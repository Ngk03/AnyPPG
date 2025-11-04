# AnyPPG
A PPG Foundation Model for Comprehensive Assessment of Multi-organ Health

```python
import torch
import torch.nn as nn
from model.net1d import Net1D

# AnyPPG encoder
anyppg_cfg = {
    in_channels: 1,
    base_filters: 64,
    ratio: 1.0,
    filter_list: [64, 160, 160, 400, 400, 1024],
    m_blocks_list: [2, 2, 2, 3, 3, 1],
    kernel_size: 3,  
    stride: 2,
    groups_width: 16,
    verbose: False,
}
anyppg = Net1D(**anyppg_cfg)

# Load pretrained model weights
ckpt_path = './checkpoint/anyppg_ckpt.pth'
params = torch.load(ckpt_path, map_location='cpu')
anyppg.load_state_dict(params)
