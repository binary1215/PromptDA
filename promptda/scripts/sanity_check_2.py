from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth
import os
from tqdm import tqdm
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


# DEVICE = 'cuda'
# image_path = "../../assets/example_images/image.jpg"
# prompt_depth_path = "../../assets/example_images/arkit_depth.png"
# image = load_image(image_path).to(DEVICE)
# prompt_depth = load_depth(prompt_depth_path).to(DEVICE) # 192x256, ARKit LiDAR depth in meters
#
# model = PromptDA.from_pretrained("../../checkpoints/Prompt-Depth-Anything-Large.ckpt").to(DEVICE).eval()
# depth = model.predict(image, prompt_depth) # HxW, depth in meters
#
# save_depth(depth, prompt_depth=prompt_depth, image=image)

DEVICE = 'cuda'
# base_path = r"D:\OneDrive - SFA\SFA_DATA\neo_pick\MDS asan dataset"
base_path = r"../../assets/pet_data/"



# 모델 로드 (한 번만)
# model = PromptDA.from_pretrained("../../checkpoints/Prompt-Depth-Anything-Large.ckpt").to(DEVICE).eval()
model = PromptDA.from_pretrained("../../checkpoints/Prompt-Depth-Anything-Small.ckpt",model_kwargs={"encoder":"vits"}).to(DEVICE).eval()

# depth 파일들 찾기
depth_files = glob.glob(os.path.join(base_path, "*_depth.npy"))

print(f"총 {len(depth_files)}개의 depth 파일을 찾았습니다.")

# 각 파일 쌍 처리
for depth_path in tqdm(depth_files, desc="Processing files"):
    # imagename 추출 (확장자와 _depth 제거)
    imagename = os.path.basename(depth_path).replace("_depth.npy", "")

    # 대응하는 이미지 파일 경로
    image_path = os.path.join(base_path, f"{imagename}.png")


    # 이미지 파일이 존재하는지 확인
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} 파일을 찾을 수 없습니다. 스킵합니다.")
        continue

    # try:
    if True:
        # 이미지와 depth 로드
        # image = load_image(image_path).to(DEVICE)
        image = load_image(image_path,max_size=490).to(DEVICE)
        # image = F.interpolate(image,
        #     size=(546, 588),  # (height, width)
        #     mode='bilinear',
        #     align_corners=False
        # )
        prompt_depth = load_depth(depth_path).to(DEVICE)  # .npy 파일 로드

        # 예측 수행
        depth = model.predict(image, prompt_depth)  # HxW, depth in meters

        # 결과 저장 (저장 경로는 필요에 따라 수정)
        # output_path = os.path.join(base_path, f"predicted/{imagename}_predicted_depth.png")
        output_path = os.path.join(f"C:/predicted/{imagename}_predicted_depth.png")
        save_depth(depth, prompt_depth=prompt_depth, image=image, output_path=output_path,show_interactive=True)

    # except Exception as e:
    #     print(f"Error processing {imagename}: {str(e)}")
    #     continue

print("모든 파일 처리 완료!")






### todo example code

# DEVICE = 'cuda'
# # base_path = r"D:\OneDrive - SFA\SFA_DATA\neo_pick\MDS asan dataset"
# base_path = r"../../assets/example_images_2/"
#
#
# # 모델 로드 (한 번만)
# model = PromptDA.from_pretrained("../../checkpoints/Prompt-Depth-Anything-Large.ckpt").to(DEVICE).eval()
#
# # depth 파일들 찾기
# # depth_files = glob.glob(os.path.join(base_path, "*_depth.npy"))
# depth_files = glob.glob(os.path.join(base_path, "*_depth.png"))
#
# print(f"총 {len(depth_files)}개의 depth 파일을 찾았습니다.")
#
# # 각 파일 쌍 처리
# for depth_path in tqdm(depth_files, desc="Processing files"):
#     # imagename 추출 (확장자와 _depth 제거)
#     # imagename = os.path.basename(depth_path).replace("_depth.npy", "")
#     imagename = os.path.basename(depth_path).replace("_depth.png", "")
#
#     # 대응하는 이미지 파일 경로
#     # image_path = os.path.join(base_path, f"{imagename}.png")
#     image_path = os.path.join(base_path, f"{imagename}.jpg")
#
#
#     # 이미지 파일이 존재하는지 확인
#     if not os.path.exists(image_path):
#         print(f"Warning: {image_path} 파일을 찾을 수 없습니다. 스킵합니다.")
#         continue
#
#     # try:
#     if True:
#         # 이미지와 depth 로드
#         image = load_image(image_path).to(DEVICE)
#         # image = F.interpolate(image,
#         #     size=(546, 588),  # (height, width)
#         #     mode='bilinear',
#         #     align_corners=False
#         # )
#         prompt_depth = load_depth(depth_path).to(DEVICE)  # .npy 파일 로드
#
#         # 예측 수행
#         depth = model.predict(image, prompt_depth)  # HxW, depth in meters
#
#         # 결과 저장 (저장 경로는 필요에 따라 수정)
#         # output_path = os.path.join(base_path, f"predicted/{imagename}_predicted_depth.png")
#         output_path = os.path.join(f"C:/predicted/{imagename}_predicted_depth.png")
#         save_depth(depth, prompt_depth=prompt_depth, image=image, output_path=output_path,show_interactive=True)
#
#     # except Exception as e:
#     #     print(f"Error processing {imagename}: {str(e)}")
#     #     continue
#
# print("모든 파일 처리 완료!")