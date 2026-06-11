#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from typing import Optional, Tuple

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    """Lazy-loading Camera: image/mask/weight are loaded from disk on first access.
    All tensors stay on CPU during camera-list loading to avoid H20 CUDA allocator crash.
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy,
                 image_name, uid,
                 image_path: str,
                 resolution: Tuple[int, int],
                 gt_alpha_mask: Optional[torch.Tensor] = None,
                 semantic_mask_path: Optional[str] = None,
                 semantic_weight_path: Optional[str] = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 data_device: str = "cuda",
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # Lazy-load paths and resolution
        self._image_path = image_path
        self._resolution = resolution
        self._gt_alpha_mask = gt_alpha_mask  # small, can keep in memory if provided
        self._semantic_mask_path = semantic_mask_path
        self._semantic_weight_path = semantic_weight_path

        self._original_image = None
        self._semantic_mask = None
        self._semantic_weight = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # Matrix compute on CPU only; lazy GPU via properties
        w2c = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32).transpose(0, 1)
        proj = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        full_proj = (w2c.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        try:
            inv_w2c = torch.linalg.inv(w2c)
        except RuntimeError:
            print(f"[Warning] Singular world_view_transform for camera {image_name}, using pinv")
            inv_w2c = torch.linalg.pinv(w2c)
        cam_center = inv_w2c[3, :3]

        self._world_view_transform = w2c
        self._projection_matrix = proj
        self._full_proj_transform = full_proj
        self._camera_center = cam_center

    @property
    def image_width(self):
        return self._resolution[0]

    @property
    def image_height(self):
        return self._resolution[1]

    @property
    def width(self):
        return self._resolution[0]

    @property
    def height(self):
        return self._resolution[1]

    @property
    def original_image(self):
        if self._original_image is None:
            from PIL import Image
            from utils.general_utils import PILtoTorch
            image = Image.open(self._image_path)
            resized_image_rgb = PILtoTorch(image, self._resolution)
            gt_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if self._gt_alpha_mask is not None:
                gt_image *= self._gt_alpha_mask
            else:
                gt_image *= torch.ones((1, self.image_height, self.image_width))
            self._original_image = gt_image
        return self._original_image

    @property
    def semantic_mask(self):
        if self._semantic_mask is None and self._semantic_mask_path is not None:
            from PIL import Image
            import numpy as np
            mask_pil = Image.open(self._semantic_mask_path)
            if mask_pil.size != self._resolution:
                mask_pil = mask_pil.resize(self._resolution, Image.NEAREST)
            mask_np = np.array(mask_pil)
            mask_tensor = torch.from_numpy(mask_np)
            if len(mask_tensor.shape) == 3:
                mask_tensor = mask_tensor[:, :, 0]
            self._semantic_mask = mask_tensor.unsqueeze(0).long()
        return self._semantic_mask

    @property
    def semantic_weight(self):
        if self._semantic_weight is None and self._semantic_weight_path is not None:
            from PIL import Image
            import numpy as np
            weight_pil = Image.open(self._semantic_weight_path)
            if weight_pil.size != self._resolution:
                weight_pil = weight_pil.resize(self._resolution, Image.BILINEAR)
            weight_np = np.array(weight_pil).astype(np.float32)
            if len(weight_np.shape) == 3:
                weight_np = weight_np[:, :, 0]
            self._semantic_weight = torch.from_numpy(weight_np).unsqueeze(0) / 255.0
        return self._semantic_weight

    @property
    def world_view_transform(self):
        return self._world_view_transform.to(self.data_device)

    @property
    def projection_matrix(self):
        return self._projection_matrix.to(self.data_device)

    @property
    def full_proj_transform(self):
        return self._full_proj_transform.to(self.data_device)

    @property
    def camera_center(self):
        return self._camera_center.to(self.data_device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        self._world_view_transform = world_view_transform
        self._full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self._world_view_transform)
        self._camera_center = view_inv[3][:3]

    @property
    def world_view_transform(self):
        return self._world_view_transform.cuda()

    @property
    def full_proj_transform(self):
        return self._full_proj_transform.cuda()

    @property
    def camera_center(self):
        return self._camera_center.cuda()
