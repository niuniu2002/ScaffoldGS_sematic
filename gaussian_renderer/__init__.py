#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Extended for 3D semantic segmentation (see PROJECT.md for details).
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from einops import repeat
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    # per-anchor segmentation confidence
    # Detach to avoid segmentation supervision pulling shared features (which can hurt RGB reconstruction).
    # 兼容 JIT traced 模型：JIT trace 只捕获默认 forward 路径，不支持 return_logit 参数。
    if getattr(pc, 'mlp_segmentation_is_jit', False):
        segmentation_anchor = pc.mlp_segmentation(feat.detach())  # [N, C] prob
        if pc.num_classes == 1:
            segmentation_anchor_logit = torch.logit(segmentation_anchor.clamp(min=1e-6, max=1-1e-6))
        else:
            segmentation_anchor_logit = torch.log(segmentation_anchor.clamp(min=1e-7))
    else:
        if pc.num_classes == 1:
            segmentation_anchor = pc.mlp_segmentation(feat.detach())  # [N, 1] prob
            segmentation_anchor_logit = torch.logit(segmentation_anchor.clamp(min=1e-6, max=1-1e-6))
        else:
            segmentation_anchor_logit = pc.mlp_segmentation(feat.detach(), return_logit=True)  # [N, C] logit
            segmentation_anchor = torch.softmax(segmentation_anchor_logit, dim=-1)

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]

    # expand per-anchor segmentation to per-Gaussian
    num_seg_ch = pc.num_classes if pc.num_classes > 1 else 1
    if pc.use_per_gaussian_seg:
        if pc.num_classes > 1:
            # [N, num_classes*n_offsets] -> [N, n_offsets, num_classes] -> [N*k, num_classes]
            segmentation_all = segmentation_anchor.view(anchor.shape[0], pc.n_offsets, pc.num_classes).view(-1, pc.num_classes)
            seg_logits_all = segmentation_anchor_logit.view(anchor.shape[0], pc.n_offsets, pc.num_classes).view(-1, pc.num_classes)
        else:
            segmentation_all = segmentation_anchor.view(-1, 1)       # [N*k, 1]
            seg_logits_all = segmentation_anchor_logit.view(-1, 1)   # [N*k, 1]
    else:
        segmentation_all = segmentation_anchor.repeat_interleave(pc.n_offsets, dim=0)       # [N*k, C]
        seg_logits_all = segmentation_anchor_logit.repeat_interleave(pc.n_offsets, dim=0)   # [N*k, C]

    # anchor index for each child Gaussian (before opacity masking)
    anchor_idx_all = torch.arange(anchor.shape[0], device=anchor.device).repeat_interleave(pc.n_offsets)  # [N*k]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, segmentation_all], dim=-1)
    masked = concatenated_all[mask]
    seg_ch = pc.num_classes if pc.num_classes > 1 else 1
    scaling_repeat, repeat_anchor, color, scale_rot, offsets, segmentation = masked.split([6, 3, 3, 7, 3, seg_ch], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) 
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    # mask seg logits and anchor indices for region consistency loss
    seg_logits_masked = seg_logits_all[mask]      # [M, 1]
    anchor_idx_masked = anchor_idx_all[mask]       # [M]

    return xyz, color, opacity, scaling, rot, neural_opacity, mask, segmentation, seg_logits_masked, anchor_idx_masked

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, iteration=-1, opacity_grad_until=-1, skip_mask=False):
    """
    Render the scene.
    skip_mask: if True, skip the 128-channel semantic mask pass (saves ~50% time during geometry-only phase).
    """
    is_training = pc.get_color_mlp.training
        
    # [修改点2] 始终接收完整返回值
    (xyz, color, opacity, scaling, rot, neural_opacity, mask, segmentation,
     seg_logits_masked, anchor_idx_masked) = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)

    # Create zero tensor.
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image
    # [Feature-3DGS rasterizer] returns (color, feature_map, radii, depth)
    rendered_image, _, radii, _ = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    # [修改点3] 语义掩码光栅化
    # skip_mask=True 时跳过整个 mask pass（纯几何训练阶段加速）
    if skip_mask:
        return {"render": rendered_image,
                "mask": None,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "opacity": opacity}

    # NOTE: Mask pass uses a black background so pixels with no Gaussian coverage
    # render to 0 (matches typical GT masks), independent of dataset white_background.
    # [Feature-3DGS rasterizer] semantic_feature is rendered via its own multi-channel path.
    raster_settings_mask = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer_mask = GaussianRasterizer(raster_settings=raster_settings_mask)

    # [关键改进 v5] 使用 semantic_feature 通道直接渲染 mask，不再需要复制3通道的 hack。
    # segmentation: [M, 1] (binary) or [M, num_classes] (multi-class)
    # 多分类时：rasterizer 必须传入 probability（alpha-blend 对 logits 无意义，
    # weighted average of logits != logit of weighted average）。
    # 二分类时：保持传入 probability（与现有 BCE/focal loss 兼容）。
    # CUDA rasterizer 要求 semantic_feature 的通道数必须严格等于 NUM_SEMANTIC_CHANNELS (128)，
    # 因此需要将 [M, C] pad 到 [M, 128]，渲染后再切回前 C 个通道。
    NUM_SEMANTIC_CHANNELS = 128
    if pc.num_classes == 1:
        seg_feature = segmentation  # [M, 1] probabilities
    else:
        seg_feature = torch.softmax(seg_logits_masked, dim=-1)  # [M, C] probabilities
    # 保存原始的通道数，用于后续切片（padding 后 shape 会变为 128）
    orig_seg_ch = seg_feature.shape[1]
    if seg_feature.shape[1] < NUM_SEMANTIC_CHANNELS:
        pad = torch.zeros(seg_feature.shape[0], NUM_SEMANTIC_CHANNELS - seg_feature.shape[1],
                          device=seg_feature.device, dtype=seg_feature.dtype)
        seg_feature = torch.cat([seg_feature, pad], dim=1)
    # [关键改进 v4] 只 detach opacity：
    # 切断 mask loss → opacity → mlp_opacity → feat → RGB 的主要冲突路径。
    # 保留 xyz/scaling/rotation 的梯度，让 mask 仍可通过调整位置/大小来对齐 GT。
    # --no_opacity_detach 时不做 detach，允许 mask loss 流向 opacity MLP（消融用）
    # --opacity_grad_until > 0 时：iteration < threshold 则允许梯度，之后 detach（动态门控）
    use_no_detach = getattr(pc, 'no_opacity_detach', False)
    if use_no_detach and opacity_grad_until > 0 and iteration >= 0:
        allow_grad = iteration < opacity_grad_until
        mask_opacities = opacity if allow_grad else opacity.detach()
    else:
        mask_opacities = opacity if use_no_detach else opacity.detach()
    _, rendered_mask_feature, _, _ = rasterizer_mask(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = torch.zeros_like(color),  # dummy colors for mask pass
        semantic_feature = seg_feature,
        opacities = mask_opacities,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None,
    )
    # rendered_mask_feature: [NUM_SEMANTIC_CHANNELS, H, W]
    # For binary: take the first channel [1, H, W]
    # For multi-class: take all C channels [C, H, W]
    if orig_seg_ch == 1:
        rendered_mask = rendered_mask_feature[0:1, :, :]
    else:
        rendered_mask = rendered_mask_feature[:orig_seg_ch, :, :]

    # [修改点4] 始终返回包含 mask 的完整字典
    return {"render": rendered_image,
            "mask": rendered_mask,  # [1, H, W] for binary or [C, H, W] for multi-class
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "neural_opacity_filtered": opacity,   # [M, 1]   mask后的child Gaussian opacity
            "scaling": scaling,
            "segmentation": segmentation,
            "neural_xyz": xyz,                    # [M, 3]   child Gaussian world coords
            "neural_seg_logits": seg_logits_masked,  # [M, 1]   pre-sigmoid seg logits
            "neural_anchor_idx": anchor_idx_masked,   # [M]      anchor index per child
            }

def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    # ... (这部分保持原样即可)
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0