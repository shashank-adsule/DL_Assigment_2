"""
report_2_4_feature_maps.py
===========================
W&B Report Section 2.4 — Inside the Black Box: Feature Maps

What this script does:
  1. Loads a trained VGG11 checkpoint.
  2. Picks a single dog image from the val set.
  3. Extracts feature maps from:
       - First conv layer  (Block 1, learns edges / colours)
       - Last conv layer before final pool  (Block 5 conv 2, learns semantics)
  4. Logs a grid of the first 32 channels for each layer to W&B.
  5. Also logs a matplotlib side-by-side comparison figure.

Run:
    python report_2_4_feature_maps.py \
        --data_root   /path/to/oxford_pets \
        --vgg11_ckpt  outputs/vgg11_bn1_dp0.5_best.pt \
        --wandb_project da6401_a2_report
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wandb

from torchvision.utils import make_grid
from models.vgg11  import VGG11Encoder
from data.dataset  import get_dataloaders


# ---------------------------------------------------------------------------
def extract_feature_maps(model, image, layer_index):
    """Return feature maps (C, H, W) from a specific features layer."""
    captured = {}
    def hook(m, inp, out):
        captured['feat'] = out.detach().cpu()
    h = list(model.features.children())[layer_index].register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(image)
    h.remove()
    return captured['feat'][0]   # (C, H, W)


def make_feat_grid(feat_maps, n_channels=32, nrow=8):
    """Normalise and tile first n_channels of feature maps into a grid image."""
    maps = feat_maps[:n_channels].unsqueeze(1)          # (N, 1, H, W)
    # Normalise each channel independently to [0,1]
    mn = maps.flatten(2).min(2).values[..., None, None]
    mx = maps.flatten(2).max(2).values[..., None, None]
    maps = (maps - mn) / (mx - mn + 1e-6)
    grid = make_grid(maps, nrow=nrow, padding=2, pad_value=0.5)
    return grid.permute(1, 2, 0).numpy()               # (H, W, 1)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--vgg11_ckpt',    required=True)
    parser.add_argument('--num_channels',  type=int, default=32,
                        help='How many feature channels to display')
    parser.add_argument('--wandb_project', default='da6401_a2_report')
    parser.add_argument('--num_workers',   type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project=args.wandb_project, name='2.4_feature_maps',
               config=vars(args))

    # --- Load model ---
    model = VGG11Encoder(num_classes=37)
    ckpt  = torch.load(args.vgg11_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    print(f'Loaded model from {args.vgg11_ckpt}')

    # --- Grab one dog image from val set ---
    _, val_loader, _ = get_dataloaders(
        root=args.data_root, task='classification',
        batch_size=16, num_workers=args.num_workers)

    probe_img, probe_label = None, None
    for imgs, labels in val_loader:
        # Try to find a dog (class index varies — just take first image)
        probe_img   = imgs[:1].to(device)
        probe_label = labels[0].item()
        break

    # --- Extract feature maps ---
    # First conv layer = index 0 in model.features
    # Last conv before final pool = index 26 (Block 5, second conv, before MaxPool at 27)
    first_feat = extract_feature_maps(model, probe_img, layer_index=0)
    last_feat  = extract_feature_maps(model, probe_img, layer_index=26)

    print(f'First conv feature maps: {tuple(first_feat.shape)}')
    print(f'Last  conv feature maps: {tuple(last_feat.shape)}')

    n = min(args.num_channels, first_feat.shape[0], last_feat.shape[0])

    first_grid = make_feat_grid(first_feat, n_channels=n, nrow=8)
    last_grid  = make_feat_grid(last_feat,  n_channels=n, nrow=8)

    # --- Matplotlib figure ---
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    orig_img = (probe_img.cpu().squeeze() * std + mean).clamp(0,1).permute(1,2,0).numpy()

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(orig_img)
    ax0.set_title('Input image', fontsize=13)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(first_grid[..., 0], cmap='viridis')
    ax1.set_title(f'Block 1 Conv1 — first {n} channels\n'
                  f'(edges, colours, low-level textures)', fontsize=11)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(last_grid[..., 0], cmap='magma')
    ax2.set_title(f'Block 5 Conv2 — first {n} channels\n'
                  f'(semantic shapes, snouts, ears, fur patterns)', fontsize=11)
    ax2.axis('off')

    fig.suptitle('Feature map comparison: low-level vs high-level representations',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    wandb.log({
        'input_image':             wandb.Image(orig_img),
        'feature_maps/block1_conv1': wandb.Image(first_grid[..., 0],
                                                  caption='Block 1 — edges & colours'),
        'feature_maps/block5_conv2': wandb.Image(last_grid[..., 0],
                                                  caption='Block 5 — semantic shapes'),
        'feature_maps/comparison_figure': wandb.Image(fig),
    })
    plt.close(fig)

    # Also log individual channel images for the first layer (useful in report)
    first_imgs = []
    for i in range(min(16, first_feat.shape[0])):
        ch = first_feat[i].numpy()
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
        first_imgs.append(wandb.Image(ch, caption=f'Ch {i}'))

    wandb.log({'feature_maps/block1_individual_channels': first_imgs})

    print('Section 2.4 complete. Feature maps logged to W&B.')
    wandb.finish()


if __name__ == '__main__':
    main()
