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

from argparse import ArgumentParser, Namespace
import sys
import os


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=str2bool, nargs='?', const=True)
                else:
                    # Compatibility: allow `--r` as an alias for `--resolution`.
                    # The original code already supports `--resolution` and `-r`.
                    if key == "resolution":
                        group.add_argument("--" + key, ("-" + key[0:1]), ("--" + key[0:1]), default=value, type=t)
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, type=str2bool, nargs='?', const=True)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False
        self.use_per_gaussian_seg = False
        self.num_classes = 1  # 1=binary(sigmoid), >=2=multi-class(softmax)
        self.no_opacity_detach = False  # True=mask loss can flow to opacity MLP (ablation)
        self.dual_feature = False       # True=seg branch uses independent _anchor_feat_seg instead of feat.detach()

        # --- Option A: low-dimensional rendered semantic features + 2D decoder ---
        # 0 = legacy mode (SegmentationHead outputs probabilities, pad to NUM_SEMANTIC_CHANNELS)
        # >0 = render low-dim features then decode to logits with a lightweight 2D head
        self.seg_feature_dim = 8
        self.seg_decoder_hidden = 64
        self.seg_decoder_layers = 2
        self.mask_mode = "auto"  # "auto" | "binary" | "multiclass". How to interpret mask values.
                                 # auto: detect 0/255 -> binary; binary: force 0/255 -> 0/1; multiclass: keep as-is

        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2

        # --- Scaling regularization (optional) ---
        # Weight decays exponentially from scaling_reg_start -> scaling_reg_end over the full training.
        # Helps preserve thin structures (stems/leaves) while keeping early stability.
        # Default keeps training scale stable (no decay) while using a weaker baseline.
        self.scaling_reg_start = 0.001
        self.scaling_reg_end = 0.001

        # --- Segmentation / regularization weights (optional) ---
        # Hard semantic warmup: before this iteration, skip Mask/KNN losses entirely.
        self.start_semantic_iter = 7000

        # Mask supervision weight (BCE on rendered mask vs semantic_mask)
        self.mask_weight = 0.01
        # Linearly ramp mask_weight from 0 -> mask_weight
        self.mask_warmup = 0          # iterations to keep mask weight at 0
        self.mask_ramp = 0            # iterations to linearly ramp to full weight
        # Compute mask loss every N iterations; 1 = every iteration (default)
        self.mask_every = 1

        # KNN consistency (smooth segmentation logits on anchors)
        self.knn_weight = 0.05
        # Compute KNN loss every N iterations; set <=0 to disable
        self.knn_every = 100
        # Offset for scheduling within the period (keeps old default behavior)
        self.knn_offset = 55
        # Linearly ramp knn_weight from 0 -> knn_weight
        self.knn_warmup = 0
        self.knn_ramp = 0
        # Adaptive KNN: reduce sampling and frequency in late stage to speed up training
        self.knn_adaptive = False
        self.knn_late_stage_iter = 10_000
        self.knn_late_stage_factor = 5
        self.knn_min_samples = 256
        self.knn_max_samples = 1024

        # --- Focal loss / uncertainty-aware weighting (optional) ---
        # Focal loss alpha (class imbalance)
        self.focal_alpha = 0.25
        # Focal loss gamma (hard example mining)
        self.focal_gamma = 2.0
        # Minimum pixel weight for uncertain pseudo-labels
        self.uncertainty_min = 0.1
        # Shape of the confidence-to-weight mapping; >1 further downweights uncertain (edge) pixels.
        self.uncertainty_power = 2.0

        # --- Semantic weight map (optional) ---
        # 是否启用外部语义模型生成的语义权重图
        self.use_semantic_weight = False
        # 权重策略: "hard" = 三段式硬阈值, "smooth" = 连续平滑函数
        self.sem_weight_strategy = "hard"
        # 高置信度阈值: abs_conf >= 该值视为高置信度，权重=1.0
        self.sem_weight_high = 0.7
        # 低置信度阈值: abs_conf < 该值视为低置信度，权重=0.0（忽略）
        self.sem_weight_low = 0.1
        # 中等置信度区域权重提升倍数（着重关照）
        self.sem_weight_boost = 2.0
        # smooth 策略下峰值位置 (0~0.5)
        self.sem_weight_smooth_peak = 0.5
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002
        # DroneSplat-style dynamic densify threshold scheduling
        self.schedule_densify_grad_threshold = False
        self.densify_grad_threshold_final = 0.001

        # --- Foreground-only Anchor Voting Loss ---
        self.anchor_fg_weight = 0.0
        self.anchor_fg_start_iter = 5000
        self.anchor_fg_every = 10
        self.anchor_fg_ratio_thr = 0.3
        self.anchor_fg_max_samples = 4096
        self.anchor_fg_detach_xyz = True
        self.anchor_fg_ramp = 1000

        # --- Anchor-level Soft Foreground Curve (exp08) ---
        self.anchor_soft_enable = False
        self.anchor_soft_weight = 0.002
        self.anchor_soft_tau = 0.35
        self.anchor_soft_temp = 0.10
        self.anchor_soft_min_ratio = 0.10
        self.anchor_soft_start_iter = 5000
        self.anchor_soft_ramp = 1000
        self.anchor_soft_decay_start = 10000
        self.anchor_soft_decay_end = 15000
        self.anchor_soft_max_samples = 4096

        # --- Mask weight decay schedule (optional) ---
        # When > 0, linearly decay mask_weight from mask_weight -> mask_weight_final
        # between mask_decay_start and mask_decay_end. Default -1 = disabled (no decay).
        self.mask_weight_final = -1.0
        self.mask_decay_start = -1
        self.mask_decay_end = -1

        # --- Opacity gradient gating (optional) ---
        # When > 0, allow mask loss -> opacity MLP gradients ONLY when
        # iteration < opacity_grad_until. After that, detach opacity in mask pass.
        # Default -1 = disabled (use no_opacity_detach flag as before).
        self.opacity_grad_until = -1

        # Two-stage training: only optimize segmentation head
        self.seg_only = False
        # Two-stage 模式下保留原始 seg head 权重（不重新初始化深层 MLP）
        self.seg_only_reuse_head = False

        # --- Auto Two-Stage Training (geometry pretraining + semantic fine-tuning) ---
        self.auto_twostage = False
        self.geometry_stage_iters = 30_000
        self.geometry_eval_interval = 1_000
        self.geometry_patience = 5
        self.geometry_min_delta = 0.03
        self.geometry_tie_psnr_delta = 0.1
        self.semantic_stage_iters = 30_000
        self.semantic_update_until = 0
        self.save_best_geometry = True
        self.best_geometry_dir = "best_geometry"

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
