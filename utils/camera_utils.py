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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
import torch
from tqdm import tqdm
import os

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # Lazy loading: pass paths/resolution only.  Actual PIL resize and tensor creation
    # happen on first property access inside Camera to avoid H20 CUDA allocator crash
    # during bulk camera list loading.
    semantic_mask_path = None
    if getattr(cam_info, "semantic_mask", None) is not None:
        scene_root = os.path.dirname(os.path.dirname(cam_info.image_path))
        mask_path_png = os.path.join(scene_root, "masks", cam_info.image_name + ".png")
        mask_path_jpg = os.path.join(scene_root, "masks", cam_info.image_name + ".jpg")
        semantic_mask_path = mask_path_png if os.path.exists(mask_path_png) else mask_path_jpg
        if not os.path.exists(semantic_mask_path):
            semantic_mask_path = None

    semantic_weight_path = None
    if getattr(cam_info, "semantic_weight", None) is not None:
        scene_root = os.path.dirname(os.path.dirname(cam_info.image_path))
        weight_path_png = os.path.join(scene_root, "semantic_weights", cam_info.image_name + ".png")
        weight_path_jpg = os.path.join(scene_root, "semantic_weights", cam_info.image_name + ".jpg")
        semantic_weight_path = weight_path_png if os.path.exists(weight_path_png) else weight_path_jpg
        if not os.path.exists(semantic_weight_path):
            semantic_weight_path = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_name=cam_info.image_name, uid=id,
                  image_path=cam_info.image_path,
                  resolution=resolution,
                  data_device=args.data_device,
                  semantic_mask_path=semantic_mask_path,
                  semantic_weight_path=semantic_weight_path)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos, desc="Loading Cameras")):
        try:
            cam = loadCam(args, id, c, resolution_scale)
            camera_list.append(cam)
        except Exception as e:
            print(f"\n[ERROR] Failed to load camera {id} ({c.image_name}): {e}")
            raise

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
