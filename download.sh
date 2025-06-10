#!/bin/bash

#download open.zip
wget https://cfiles.dacon.co.kr/comprtitions/236493/open.zip
python3 - << 'EOF'
import shutil
shutil.unpack_archive("open.zip","./","zip")
EOF

#download lib
pip install -q pandas timm transformers scikit-learn oprn_clip_torch GPUtil gpustat wandb

wandb login 63e8a68f421abc6880201105ab26f59bbd0fla87

