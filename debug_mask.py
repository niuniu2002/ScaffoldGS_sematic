import torch
import sys
import os
sys.path.insert(0, '.')
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
import argparse

parser = argparse.ArgumentParser()
ModelParams(parser)
args = parser.parse_args(['--source_path', 'data/dronev4_2', '--model_path', 'output/dronev4_2', '--eval', '--appearance_dim', '0'])

num_classes = getattr(args, 'num_classes', 1)
gaussians = GaussianModel(args.feat_dim, args.n_offsets, args.voxel_size, args.update_depth, args.update_init_factor, args.update_hierachy_factor, args.use_feat_bank, args.appearance_dim, args.ratio, args.add_opacity_dist, args.add_cov_dist, args.add_color_dist, num_classes=num_classes)
if num_classes > 1:
    print(f"[WARNING] debug_mask.py currently only supports binary segmentation (num_classes=1). num_classes={num_classes} is set, but IoU/threshold logic assumes binary. Results may be misleading.")

# Monkey-patch to skip automatic mlp loading
gaussians.load_mlp_checkpoints = lambda *a, **k: None

scene = Scene(args, gaussians, load_iteration=30000, shuffle=False)

# Load MLPs with jit (split mode)
pc_dir = 'output/dronev4_2/point_cloud/iteration_30000'
gaussians.mlp_opacity = torch.jit.load(os.path.join(pc_dir, 'opacity_mlp.pt')).cuda()
gaussians.mlp_cov = torch.jit.load(os.path.join(pc_dir, 'cov_mlp.pt')).cuda()
gaussians.mlp_color = torch.jit.load(os.path.join(pc_dir, 'color_mlp.pt')).cuda()
gaussians.mlp_segmentation = torch.jit.load(os.path.join(pc_dir, 'segmentation_mlp.pt')).cuda()

viewpoint = scene.getTestCameras()[0]
print('Camera:', viewpoint.image_name)

bg_color = [1,1,1] if args.white_background else [0,0,0]
background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

class FakePipe:
    def __init__(self):
        self.debug = False
        self.compute_cov3D_python = False
        self.convert_SHs_python = False
pipe = FakePipe()

render_pkg = render(viewpoint, gaussians, pipe, background)
mask = render_pkg['mask']
print('Mask shape:', mask.shape)
print('Mask min/max/mean:', mask.min().item(), mask.max().item(), mask.mean().item())
print('Mask > 0.5 count:', (mask > 0.5).sum().item(), 'out of', mask.numel())
print('Mask percentiles:')
for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f'  {p}%: {torch.quantile(mask.flatten(), p/100).item():.6f}')

if hasattr(viewpoint, 'semantic_mask') and viewpoint.semantic_mask is not None:
    gt = viewpoint.semantic_mask.cuda()
    print('GT mask shape:', gt.shape)
    print('GT min/max/mean:', gt.min().item(), gt.max().item(), gt.mean().item())
    print('GT > 0.5 count:', (gt > 0.5).sum().item(), 'out of', gt.numel())
    
    p_mask = (mask > 0.5).float()
    g_mask = (gt > 0.5).float()
    inter = (p_mask * g_mask).sum().item()
    union = p_mask.sum().item() + g_mask.sum().item() - inter
    iou = inter / union
    print(f'IoU on this image: {iou:.4f}')
    
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p_mask_t = (mask > thresh).float()
        inter_t = (p_mask_t * g_mask).sum().item()
        union_t = p_mask_t.sum().item() + g_mask.sum().item() - inter_t
        iou_t = inter_t / union_t
        print(f'  IoU @ thresh={thresh}: {iou_t:.4f}')
