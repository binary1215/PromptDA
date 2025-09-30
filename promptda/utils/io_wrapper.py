import numpy as np
import imageio
import torch
import os
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from promptda.utils.logger import Log
from promptda.utils.depth_utils import visualize_depth
from promptda.utils.parallel_utils import async_call

# DEVICE = 'cuda' if torch.cuda.is_available(
# ) else 'mps' if torch.backends.mps.is_available() else 'cpu'


def to_tensor_func(arr):
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_numpy_func(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def ensure_multiple_of(x, multiple_of=14):
    return int(x // multiple_of * multiple_of)


def load_image(image_path, to_tensor=True, max_size=1008, multiple_of=14):
    '''
    Load image from path and convert to tensor
    max_size // 14 = 0
    '''
    image = np.asarray(imageio.imread(image_path)).astype(np.float32)
    image = image / 255.

    max_size = max_size // multiple_of * multiple_of
    if max(image.shape) > max_size:
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        tar_h = ensure_multiple_of(h * scale)
        tar_w = ensure_multiple_of(w * scale)
        image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)
    if to_tensor:
        return to_tensor_func(image)
    return image


def load_depth(depth_path, to_tensor=True):
    '''
    Load depth from path and convert to tensor
    '''
    if depth_path.endswith('.png'):
        depth = np.asarray(imageio.imread(depth_path)).astype(np.float32)
        depth = depth / 1000.
    elif depth_path.endswith('.npz'):
        depth = np.load(depth_path)['depth']
    elif depth_path.endswith('.npy'):
        depth = np.load(depth_path).astype(np.float32)
        depth = depth / 1000.
    else:
        raise ValueError(f"Unsupported depth format: {depth_path}")
    if to_tensor:
        return to_tensor_func(depth)
    return depth

#
# # @async_call
# def save_depth(depth,
#                prompt_depth=None,
#                image=None,
#                output_path='results/example_depth.png',
#                save_vis=True):
#     '''
#     Save depth to path
#     '''
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     depth = to_numpy_func(depth)
#     uint16_depth = (depth).astype(np.uint16)
#     imageio.imwrite(output_path, uint16_depth)
#     Log.info(f'Saved depth to {output_path}', tag='save_depth')
#
#     if not save_vis:
#         return
#     output_path_ = output_path
#     output_path = output_path_.replace('.png', '_depth.jpg')
#     depth_vis, depth_min, depth_max = visualize_depth(depth, ret_minmax=True)
#     imageio.imwrite(output_path, depth_vis)
#
#
#     if prompt_depth is not None:
#         prompt_depth = to_numpy_func(prompt_depth)
#         output_path = output_path_.replace('.png', '_prompt_depth.jpg')
#         prompt_depth_vis = visualize_depth(prompt_depth,
#                                            depth_min=depth_min,
#                                            depth_max=depth_max)
#         imageio.imwrite(output_path, prompt_depth_vis)
#
#     if image is not None:
#         output_path = output_path_.replace('.png', '_image.jpg')
#         image = to_numpy_func(image)
#         imageio.imwrite(output_path, (image * 255).astype(np.uint8))

def visualize_depth_with_image(depth, image=None, window_name="Depth Visualization"):
    '''
    depth map과 이미지를 함께 시각화 (이미지는 matplotlib, 3D는 open3d)

    Args:
        depth: (H, W) depth array
        image: (H, W, 3) RGB image
        prompt_depth: (H, W) prompt depth (optional)
        window_name: visualization window name
    '''
    # RGB 이미지 준비
    rgb_image = None
    if image is not None:
        if image.ndim == 3 and image.shape[0]>3:
            rgb_image = image
        else:
            rgb_image = to_numpy_func(image)
            if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:  # (C, H, W)
                rgb_image = rgb_image.transpose(1, 2, 0)  # (H, W, C)
            if rgb_image.ndim == 4 and rgb_image.shape[1] == 3 and rgb_image.shape[0] == 1:  # (B, C, H, W)
                rgb_image = rgb_image[0].transpose(1, 2, 0)  # (H, W, C)
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
    #
    # # Matplotlib으로 2D 이미지들 표시
    # n_plots = 1 + (image is not None) + (prompt_depth is not None)
    # fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    # if n_plots == 1:
    #     axes = [axes]
    #
    # plot_idx = 0
    #
    # # 원본 이미지
    # if image is not None:
    #     axes[plot_idx].imshow(rgb_image)
    #     axes[plot_idx].set_title('Original Image')
    #     axes[plot_idx].axis('off')
    #     plot_idx += 1
    #
    # # Depth 시각화
    # depth_vis = visualize_depth(to_numpy_func(depth))
    # axes[plot_idx].imshow(depth_vis)
    # axes[plot_idx].set_title('Predicted Depth')
    # axes[plot_idx].axis('off')
    # plot_idx += 1
    #
    # # Prompt depth
    # if prompt_depth is not None:
    #     prompt_depth_vis = visualize_depth(to_numpy_func(prompt_depth))
    #     axes[plot_idx].imshow(prompt_depth_vis)
    #     axes[plot_idx].set_title('Prompt Depth')
    #     axes[plot_idx].axis('off')
    #
    # plt.tight_layout()
    # plt.show(block=False)  # non-blocking

    # Point cloud 생성 및 시각화
    # depth_np = to_numpy_func(depth)
    if torch.is_tensor(depth): depth = depth.cpu().numpy()
    if depth.ndim > 2 and depth.shape[0] == 1:depth = (np.squeeze(depth)*1000.).astype(np.uint16)
    if image.shape[:2] != depth.shape[:2]:
        image = cv2.resize(image, depth.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        # image = np.resize(image,  (*depth.shape[:2],3))
    pcd = create_point_cloud_from_depth(np.squeeze(depth), image=image)

    # Open3D 시각화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1000, height=800)
    vis.add_geometry(pcd)

    # 시점 설정
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, -1, 0])

    # 렌더링 옵션
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()



def save_depth(depth,
               prompt_depth=None,
               image=None,
               output_path='results/example_depth.png',
               save_vis=True,
               save_3d=True,
               show_interactive=False):  # 대화형 시각화 옵션
    '''
    Save depth to path with optional 3D visualization
    '''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    depth = to_numpy_func(depth)
    uint16_depth = (depth * 1000.).astype(np.uint16)
    imageio.imwrite(output_path, uint16_depth)
    Log.info(f'Saved depth to {output_path}', tag='save_depth')

    if not save_vis:
        return

    output_path_ = output_path
    output_path = output_path_.replace('.png', '_depth.jpg')
    depth_vis, depth_min, depth_max = visualize_depth(depth, ret_minmax=True)
    imageio.imwrite(output_path, depth_vis)

    if prompt_depth is not None:
        prompt_depth_np = to_numpy_func(prompt_depth)
        output_path = output_path_.replace('.png', '_prompt_depth.jpg')
        prompt_depth_vis = visualize_depth(prompt_depth_np,
                                           depth_min=depth_min,
                                           depth_max=depth_max)
        imageio.imwrite(output_path, prompt_depth_vis)

    if image is not None:
        output_path = output_path_.replace('.png', '_image.jpg')
        image_np = to_numpy_func(image)
        imageio.imwrite(output_path, (image_np * 255).astype(np.uint8))

    # 3D point cloud 저장
    if save_3d:
        rgb_image = None
        if image is not None:
            rgb_image = to_numpy_func(image)
            if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:
                rgb_image = rgb_image.transpose(1, 2, 0)

        pcd = create_point_cloud_from_depth(depth, image=rgb_image)
        ply_path = output_path_.replace('.png', '_pointcloud.ply')
        o3d.io.write_point_cloud(ply_path, pcd)
        Log.info(f'Saved point cloud to {ply_path}', tag='save_depth')

    # 대화형 시각화
    if show_interactive:
        # visualize_depth_with_image(depth, image=rgb_image, prompt_depth=prompt_depth,
        #                            window_name=os.path.basename(output_path_))
        visualize_depth_with_image(depth, image=rgb_image,
                                   window_name="pred_depth")
        visualize_depth_with_image(prompt_depth, image=rgb_image,
                                   window_name="original_depth")


def create_point_cloud_from_depth(depth, image=None, fx=None, fy=None, cx=None, cy=None):
    '''
    depth map에서 point cloud 생성
    '''
    h, w = depth.shape

    if fx is None:
        fx = w
    if fy is None:
        fy = h
    if cx is None:
        cx = w / 2
    if cy is None:
        cy = h / 2

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    valid = (z > 0) & (z < np.percentile(z[z > 0], 99))
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if image is not None:
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        colors = image[valid].reshape(-1, 3)
        if colors.max() <= 1.0:
            colors = colors
        else:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd