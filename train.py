
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
import timm
from dataload import get_loaders, CFG
from dataload import seed_everything

# Model definition
class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = timm.create_model(
            'swin_large_patch4_window12_384',
            pretrained=True,
            num_classes=num_classes
        )
    def forward(self, x):
        return self.backbone(x)

# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        at = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
        fl = at * (1 - pt) ** self.gamma * ce_loss
        return fl.mean() if self.reduction == 'mean' else fl.sum()

def train():
    run = wandb.init(project="kon")
    wandb.config.update({
        "learning_rate": CFG['LEARNING_RATE'],
        "epochs": CFG['EPOCHS'],
        "batch_size": CFG['BATCH_SIZE'],
        "optimizer": "AdamW",
        "loss_function": "FocalLoss",
        "model": "BaseModel",
    })
    seed_everything(CFG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, class_names = get_loaders()

    model = BaseModel(num_classes=len(class_names)).to(device)
    # Freeze backbone initial epochs
    freeze_epochs = 10
    for name, param in model.backbone.named_parameters():
        param.requires_grad = 'head' in name or 'classifier' in name

    optimizer = optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=0.01)
    total_steps = len(train_loader) * CFG['EPOCHS']
    warmup_steps = len(train_loader) * 3
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = GradScaler()

    # class weights for focal loss
    cls_counts = torch.tensor([len([1 for _, lbl in train_loader.dataset if lbl == i]) for i in range(len(class_names))], dtype=torch.float32)
    alpha = (len(train_loader.dataset) / (len(class_names) * cls_counts)).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    best_logloss = float('inf')
    patience = 7
    trigger_times = 0

    for epoch in range(CFG['EPOCHS']):
        if epoch == freeze_epochs:
            for name, param in model.backbone.named_parameters():
                param.requires_grad = True

        model.train()
        train_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            if (step + 1) % 10 == 0:
                wandb.log({
                    "epoch": epoch + 1,
                    "step": epoch * len(train_loader) + step + 1,
                    "lr": scheduler.get_last_lr()[0],
                    "train_loss": train_loss / (step + 1)
                })

        # Validation
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_logloss = torch.tensor(0.0)  # compute log_loss externally if needed

        if val_logloss < best_logloss:
            best_logloss = val_logloss
            torch.save(model.state_dict(), 'best_model.pth')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

        wandb.log({
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
        })

    wandb.finish()

if __name__ == "__main__":
    train()
