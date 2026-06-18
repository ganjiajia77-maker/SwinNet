import argparse
import os
import sys
import torch

# Ensure project root is on sys.path so we can import top-level modules when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import get_config
from networks.vision_transformer import SwinUnet


def make_args():
    # minimal args object compatible with get_config
    return argparse.Namespace(
        cfg='./configs/swin_tiny_patch4_window7_224_lite.yaml',
        # override YAML window size to 8 so it divides 128 (512/4)
        opts=['MODEL.SWIN.WINDOW_SIZE', '8'],
        batch_size=None,
        img_size=512,
        zip=False,
        cache_mode='',
        resume='',
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level='',
        tag='',
        eval=False,
        throughput=False,
    )


def main():
    args = make_args()
    config = get_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SwinUnet(config=config, img_size=args.img_size, num_classes=1, use_asterisk=False, return_skeleton=False)
    model.to(device)

    x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        out = model(x)

    print('Dummy forward completed.')
    if isinstance(out, torch.Tensor):
        print('Output shape:', tuple(out.shape))
    elif isinstance(out, tuple):
        shapes = []
        for o in out:
            if isinstance(o, torch.Tensor):
                shapes.append(tuple(o.shape))
            else:
                shapes.append(str(type(o)))
        print('Output tuple shapes:', shapes)


if __name__ == '__main__':
    main()
