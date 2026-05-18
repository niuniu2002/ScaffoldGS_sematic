#
# render_video.py — Camera-path video rendering for Scaffold-GS
#
# Usage:  python render_video.py -m <path_to_model> --output <out.mp4> [options]
#
# Features:
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#   - Generates smooth camera trajectories using SLERP / linear interpolation
#   - Renders frames along the path and encodes them into a video
#   - Supports orbit, spiral, and custom camera-path modes
#
# render_video_scaffold.py
# 安全模式：线性插值 (Linear/SLERP)
# 牺牲一点平滑度，换取 100% 的稳定性，防止相机飞出锚点范围

import torch
import torchvision
import numpy as np
import subprocess
import json
import time
import imageio.v2 as imageio
import cv2
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import gc

# 自动显卡选择
try:
	cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
	result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
	os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
except:
	pass

from scene import Scene
from gaussian_renderer import render, prefilter_voxel
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera 

# === 核心修改：线性插值算法 (安全模式) ===
def get_linear_trajectory(camera_list, n_frames):
	"""
	使用线性插值生成轨迹。
	保证相机始终位于训练轨迹的连线上，绝不越界。
	"""
	print(f"Generating LINEAR trajectory (Safety Mode) from {len(camera_list)} cameras...")
    
	# 1. 提取所有相机位姿 (不跳帧，保证覆盖所有细节)
	positions = []
	rotations = []
    
	# 这里我们使用全部相机，确保不漏掉任何路径
	for camera in camera_list:
		positions.append(camera.T)
		rotations.append(R.from_matrix(camera.R))
    
	positions = np.stack(positions)
    
	# 2. 准备插值
	# 我们将 n_frames 分配到各个相机段之间
	num_segments = len(camera_list) - 1
	if num_segments < 1:
		return np.array(positions), np.array([r.as_quat() for r in rotations])

	# 生成时间戳
	key_times = np.linspace(0, 1, len(camera_list))
	target_times = np.linspace(0, 1, n_frames)

	# 3. 位置插值 (Linear Interpolation)
	# 使用简单的线性插值
	pos_interp = np.zeros((n_frames, 3))
	for i in range(3):
		pos_interp[:, i] = np.interp(target_times, key_times, positions[:, i])

	# 4. 旋转插值 (SLERP - Spherical Linear Interpolation)
	rotation_spline = Slerp(key_times, R.from_matrix([c.R for c in camera_list]))
	rot_interp_obj = rotation_spline(target_times)
	rot_interp = rot_interp_obj.as_quat() # (x, y, z, w)

	return pos_interp, rot_interp

def images_to_video(image_folder, output_video_path, fps=30, max_frames=None):
	images = []
	file_list = sorted(os.listdir(image_folder))
	if not file_list:
		print("No frames found for video encoding.")
		return

	valid_files = sorted(f for f in file_list if f.endswith((".png", ".jpg")))
	if not valid_files:
		print("No valid image files (.png/.jpg) found for video encoding.")
		return

	# 如果目录中存在比本次渲染更多的旧帧，只使用前 max_frames 帧
	if max_frames is not None:
		valid_files = valid_files[:max_frames]

	print(f"Encoding video from {len(valid_files)} frames...")

	for filename in valid_files:
		path = os.path.join(image_folder, filename)
		try:
			images.append(imageio.imread(path))
		except Exception as e:
			print(f"[Warning] Failed to read frame {path}: {e}")

	if not images:
		print("No frames could be read successfully; aborting video write.")
		return

	try:
		imageio.mimwrite(output_video_path, images, fps=fps)
		print(f"Video saved to: {output_video_path}")
	except Exception as e:
		print(f"[Error] Failed to write video {output_video_path}: {e}")

# === 渲染核心 ===
def render_video_path(dataset, iteration, pipeline, args):
	with torch.no_grad():
		gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, 
								dataset.update_depth, dataset.update_init_factor, 
								dataset.update_hierachy_factor, dataset.use_feat_bank, 
								dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, 
								dataset.add_cov_dist, dataset.add_color_dist, num_classes=getattr(dataset, 'num_classes', 1))
        
		scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
		# 使用 eval 加载权重，然后切换到 train 模式以启用语义 mask 渲染
		gaussians.eval()
		gaussians.train()

		bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
		background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

		# 获取参考相机
		base_cameras = scene.getTestCameras()
		if len(base_cameras) == 0:
			print("No test cameras found, using training cameras for trajectory.")
			base_cameras = scene.getTrainCameras()
        
		# [修改] 使用线性插值
		pos_interp, rot_interp = get_linear_trajectory(base_cameras, args.n_views)

		# 准备 Dummy Image
		ref_cam = base_cameras[0]
		if hasattr(ref_cam, 'image_height'):
			H, W = ref_cam.image_height, ref_cam.image_width
		elif hasattr(ref_cam, 'original_image'):
			H, W = ref_cam.original_image.shape[1], ref_cam.original_image.shape[2]
		else:
			H, W = 1080, 1920
            
		dummy_image = torch.zeros((3, H, W), dtype=torch.float32, device="cpu")

		# 输出路径
		base_video_dir = os.path.join(dataset.model_path, "video_interp")
		render_path = os.path.join(base_video_dir, "renders")
		mask_path = os.path.join(base_video_dir, "masks")
		masked_rgb_path = os.path.join(base_video_dir, "masked_rgb")
		os.makedirs(render_path, exist_ok=True)
		os.makedirs(mask_path, exist_ok=True)
		os.makedirs(masked_rgb_path, exist_ok=True)

		print(f"Starting Linear Rendering for {args.n_views} frames...")

		# 使用一个合法的相机 uid，避免 appearance embedding 越界
		# 这里复用参考相机的 uid，而不是用新帧索引
		valid_uid = getattr(ref_cam, "uid", 0)
        
		# === 循环渲染 ===
		for idx in tqdm(range(args.n_views), desc="Rendering"):
			rgb_file = os.path.join(render_path, '{0:05d}'.format(idx) + ".png")
			mask_file = os.path.join(mask_path, '{0:05d}'.format(idx) + ".png")
			masked_rgb_file = os.path.join(masked_rgb_path, '{0:05d}'.format(idx) + ".png")

			# 断点续传：三种帧都存在就跳过
			if os.path.exists(rgb_file) and os.path.exists(mask_file) and os.path.exists(masked_rgb_file):
				continue

			try:
				# JIT 创建相机
				cur_pos = pos_interp[idx]
				cur_rot = rot_interp[idx]
				cur_R = R.from_quat(cur_rot).as_matrix()
                
				temp_cam = Camera(
					colmap_id=0,
					R=cur_R,
					T=cur_pos,
					FoVx=ref_cam.FoVx,
					FoVy=ref_cam.FoVy,
					image=dummy_image, 
					gt_alpha_mask=None,
					image_name=f"frame_{idx}",
					uid=valid_uid,
					data_device="cuda"
				)
				
				voxel_visible_mask = prefilter_voxel(temp_cam, gaussians, pipeline, background)
				
				# 尝试渲染 (兼容性写法)
				try:
					render_pkg = render(temp_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask, is_training=True)
				except TypeError:
					render_pkg = render(temp_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask)

				rendering = render_pkg["render"]
				pred_mask = render_pkg.get("mask", None)
				# 容错：如果没有 mask 则给全黑 / 或用 segmentation 兜底
				if pred_mask is None:
					if "segmentation" in render_pkg:
						pred_mask = (render_pkg["segmentation"] > 0.5).float()
					else:
						pred_mask = torch.zeros((1, H, W), dtype=torch.float32, device=rendering.device)
				else:
					pred_mask = pred_mask.clamp(0.0, 1.0)

				# === 橙色高亮覆盖 (Orange Overlay) ===
				# 橙色 RGB: [1.0, 0.5, 0.0]
				highlight_color = torch.tensor([1.0, 0.5, 0.0], device=rendering.device).view(3, 1, 1)
				blend_factor = 0.5

				# 混合公式
				mask_weight = pred_mask * blend_factor
				masked_rgb = rendering * (1.0 - mask_weight) + highlight_color * mask_weight

				# 保存图片
				torchvision.utils.save_image(rendering.cpu(), rgb_file)
				torchvision.utils.save_image(pred_mask.cpu(), mask_file)
				torchvision.utils.save_image(masked_rgb.cpu(), masked_rgb_file)
				
				del temp_cam, rendering, pred_mask, masked_rgb, render_pkg, voxel_visible_mask
				if idx % 50 == 0:
					torch.cuda.empty_cache()
                    
			except Exception as e:
				print(f"\n[Error] Frame {idx} failed: {e}")

		# 分别导出 RGB、mask 和语义高亮 RGB 的视频
		video_rgb_path = os.path.join(dataset.model_path, f"video_rgb_{iteration}.mp4")
		video_mask_path = os.path.join(dataset.model_path, f"video_mask_{iteration}.mp4")
		video_masked_rgb_path = os.path.join(dataset.model_path, f"video_masked_rgb_{iteration}.mp4")

		# 只使用当前设置的 n_views 帧进行视频编码，避免之前长序列残留影响
		images_to_video(render_path, video_rgb_path, fps=args.fps, max_frames=args.n_views)
		images_to_video(mask_path, video_mask_path, fps=args.fps, max_frames=args.n_views)
		images_to_video(masked_rgb_path, video_masked_rgb_path, fps=args.fps, max_frames=args.n_views)

if __name__ == "__main__":
	parser = ArgumentParser(description="Scaffold-GS Video Rendering")
	model = ModelParams(parser, sentinel=True)
	pipeline = PipelineParams(parser)
	parser.add_argument("--iteration", default=-1, type=int)
	parser.add_argument("--n_views", default=2400, type=int)
	parser.add_argument("--fps", default=60, type=int)
	parser.add_argument("--quiet", action="store_true")
	args = get_combined_args(parser)
    
	print("Rendering Video for: " + args.model_path)
	safe_state(args.quiet)
	render_video_path(model.extract(args), args.iteration, pipeline.extract(args), args)