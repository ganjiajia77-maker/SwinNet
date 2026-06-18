#!/usr/bin/env python
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 手动创建一个简单的配置对象
class SimpleConfig:
    def __init__(self):
        self.window_size = 7
        self.embed_dim = 96
        self.depths = [2, 2, 2, 2]
        self.num_heads = [3, 6, 12, 24]
        self.mlp_ratio = 4.0
        self.qkv_bias = True
        self.qk_scale = None
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0
        self.drop_path_rate = 0.2
        self.patch_norm = True
        self.use_checkpoint = False
        self.fused_window_process = False

from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

try:
    config = SimpleConfig()
    model = SwinTransformerSys(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=torch.nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        use_asterisk=True,
        return_skeleton=True,
    )
    model = model.to(device)
    print('[✓] 模型加载成功')
    
    # 测试前向传播
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        outputs = model(x)
        if isinstance(outputs, tuple):
            surface_logits, skeleton_logits, attn = outputs
            print(f'[✓] Surface logits: {surface_logits.shape}')
            print(f'[✓] Skeleton logits: {skeleton_logits.shape}')
            print(f'[✓] Skeleton attention: {attn.shape}')
        else:
            print(f'[✓] Output: {outputs.shape}')
    
    print('[✓✓✓] 模型测试成功！新结构正常工作 [✓✓✓]')
    
except Exception as e:
    print(f'[✗] 错误: {e}')
    import traceback
    traceback.print_exc()
