import os
import torch
import torch.nn as nn
import random
from torchvision import transforms
from data.dataset import PairDataset
from config.config import ModelConfig
from models.feature_extractor import FeatureExtractor
from models.metric_model import *
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 모델 로드
# ---------------------------
def load_model(weight_path, method='ours'):
    model_config = ModelConfig(model_name="mobilenetv3", pretrained=False, device=device)
    feat_extractor = FeatureExtractor(model_config)
    domain_extractor = FeatureExtractor(model_config)

    if method == 'ours':
        model = ContrastiveModel_ours(feat_extractor, domain_extractor).to(device)
    elif method == 'bb':
        model = ContrastiveModel_only_BB(feat_extractor).to(device)
    elif method == 'mb':
        model = ContrastiveModel_only_MB(feat_extractor).to(device)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.eval()
    return model

# ---------------------------
# 마지막 Conv 찾아서 target layer 결정
# ---------------------------
def find_last_conv_layer(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def build_cam(backbone: nn.Module):
    target = find_last_conv_layer(backbone)
    if target is None:
        return EigenCAM(model=backbone, target_layers=[backbone])
    return GradCAM(model=backbone, target_layers=[target])

# ---------------------------
# CAM 이미지 얻기
# ---------------------------
def get_cam_image(cam_extractor, input_tensor, rgb_img_01):
    grayscale_cam = cam_extractor(input_tensor=input_tensor)[0, :]
    return show_cam_on_image(rgb_img_01, grayscale_cam, use_rgb=True)

def to_rgb01(t):
    x = t.detach().cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    return x

if __name__ == "__main__":
    base_save_dir = "gradcam_results"
    os.makedirs(base_save_dir, exist_ok=True)

    # 1) 모델 로드
    model_ours = load_model('./best_weight/ours.pth', 'ours')
    model_mb   = load_model('./best_weight/mb.pth',   'mb')
    model_bb   = load_model('./best_weight/bb.pth',   'bb')

    # 2) 데이터 로드
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = PairDataset(
        image_dir="./periocular_data",
        pairs_file="./periocular_data/fold4_pairs_valid.txt",
        transform=transform
    )

    # 동일인물 index 전부 모아 랜덤 10개 추출
    positive_indices = []
    for i in range(len(dataset)):
        label = dataset[i][4]
        if int(label) == 1:
            positive_indices.append(i)
        if len(positive_indices) >= 30:
            break
    chosen_indices = random.sample(positive_indices, min(30, len(positive_indices)))

    # CAM 준비
    cam_ours_img  = build_cam(model_ours.feature_extractor.model)
    cam_ours_diff = build_cam(model_ours.domain_feature_extractor.model)
    cam_mb_img    = build_cam(model_mb.feature_extractor.model)
    cam_bb_diff   = build_cam(model_bb.feature_extractor.model)  # bb는 diff만

    # 3) 각 샘플별로 ours(x1,x2,diff), mb(x1,x2), bb(diff) 개별 저장
    for idx in chosen_indices:
        img1_t, img2_t, *_ = dataset[idx]
        x1 = img1_t.unsqueeze(0).to(device)
        x2 = img2_t.unsqueeze(0).to(device)
        diff = (x1 - x2)

        img1_rgb = to_rgb01(img1_t)
        img2_rgb = to_rgb01(img2_t)
        diff_rgb = to_rgb01(img1_t - img2_t)

        save_dir = os.path.join(base_save_dir, f"sample_{idx}")
        os.makedirs(save_dir, exist_ok=True)

        # ours
        cv2.imwrite(os.path.join(save_dir, "ours_x1.png"),
                    cv2.cvtColor(get_cam_image(cam_ours_img,  x1, img1_rgb), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "ours_x2.png"),
                    cv2.cvtColor(get_cam_image(cam_ours_img,  x2, img2_rgb), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "ours_diff.png"),
                    cv2.cvtColor(get_cam_image(cam_ours_diff, diff, diff_rgb), cv2.COLOR_RGB2BGR))

        # mb
        cv2.imwrite(os.path.join(save_dir, "mb_x1.png"),
                    cv2.cvtColor(get_cam_image(cam_mb_img, x1, img1_rgb), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "mb_x2.png"),
                    cv2.cvtColor(get_cam_image(cam_mb_img, x2, img2_rgb), cv2.COLOR_RGB2BGR))

        # bb
        cv2.imwrite(os.path.join(save_dir, "bb_diff.png"),
                    cv2.cvtColor(get_cam_image(cam_bb_diff, diff, diff_rgb), cv2.COLOR_RGB2BGR))

        print(f"Saved all images for sample {idx} in {save_dir}")


# import os
# import torch
# import torch.nn as nn
# import random
# from torchvision import transforms
# from data.dataset import PairDataset
# from config.config import ModelConfig
# from models.feature_extractor import FeatureExtractor
# from models.metric_model import *
# from pytorch_grad_cam import GradCAM, EigenCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import numpy as np
# import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---------------------------
# # 모델 로드
# # ---------------------------
# def load_model(weight_path, method='ours'):
#     model_config = ModelConfig(model_name="mobilenetv3", pretrained=False, device=device)
#     feat_extractor = FeatureExtractor(model_config)
#     domain_extractor = FeatureExtractor(model_config)

#     if method == 'ours':
#         model = ContrastiveModel_ours(feat_extractor, domain_extractor).to(device)
#     elif method == 'bb':
#         model = ContrastiveModel_only_BB(feat_extractor).to(device)
#     elif method == 'mb':
#         model = ContrastiveModel_only_MB(feat_extractor).to(device)
#     else:
#         raise ValueError(f"Unknown method: {method}")

#     model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
#     model.eval()
#     return model

# # ---------------------------
# # 마지막 Conv 찾아서 target layer 결정
# # ---------------------------
# def find_last_conv_layer(module: nn.Module):
#     last = None
#     for m in module.modules():
#         if isinstance(m, nn.Conv2d):
#             last = m
#     return last

# def build_cam(backbone: nn.Module):
#     target = find_last_conv_layer(backbone)
#     if target is None:
#         return EigenCAM(model=backbone, target_layers=[backbone])
#     return GradCAM(model=backbone, target_layers=[target])

# # ---------------------------
# # CAM 이미지 얻기
# # ---------------------------
# def get_cam_image(cam_extractor, input_tensor, rgb_img_01):
#     grayscale_cam = cam_extractor(input_tensor=input_tensor)[0, :]
#     return show_cam_on_image(rgb_img_01, grayscale_cam, use_rgb=True)

# def to_rgb01(t):
#     x = t.detach().cpu().numpy()
#     x = np.transpose(x, (1, 2, 0))
#     x = np.clip(x, 0.0, 1.0)
#     return x

# if __name__ == "__main__":
#     os.makedirs("gradcam_results", exist_ok=True)

#     # 1) 모델 로드
#     model_ours = load_model('./best_weight/ours.pth', 'ours')
#     model_mb   = load_model('./best_weight/mb.pth',   'mb')
#     model_bb   = load_model('./best_weight/bb.pth',   'bb')

#     # 2) 데이터 로드
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     dataset = PairDataset(
#         image_dir="./periocular_data",
#         pairs_file="./periocular_data/fold4_pairs_valid.txt",
#         transform=transform
#     )

#     # 동일인물 index 전부 모아 랜덤 10개 추출
#     positive_indices = [i for i in range(len(dataset)) if int(dataset[i][4].item()) == 1]
#     if len(positive_indices) < 10:
#         raise RuntimeError("동일인물 pair가 10개 미만임")
#     chosen_indices = random.sample(positive_indices, 10)

#     # CAM 준비
#     cam_ours_img  = build_cam(model_ours.feature_extractor.model)
#     cam_ours_diff = build_cam(model_ours.domain_feature_extractor.model)
#     cam_mb_img    = build_cam(model_mb.feature_extractor.model)
#     cam_bb_diff   = build_cam(model_bb.feature_extractor.model)  # bb는 diff만

#     # 3) 각 샘플별로 ours(x1,x2,diff), mb(x1,x2), bb(diff) 저장
#     for idx in chosen_indices:
#         img1_t, img2_t, *_, _, _ = dataset[idx]
#         x1 = img1_t.unsqueeze(0).to(device)
#         x2 = img2_t.unsqueeze(0).to(device)
#         diff = (x1 - x2)

#         img1_rgb = to_rgb01(img1_t)
#         img2_rgb = to_rgb01(img2_t)
#         diff_rgb = to_rgb01(img1_t - img2_t)

#         fig, axes = plt.subplots(1, 6, figsize=(18, 3))

#         # ours
#         axes[0].imshow(get_cam_image(cam_ours_img,  x1,   img1_rgb)); axes[0].set_title("ours-x1"); axes[0].axis('off')
#         axes[1].imshow(get_cam_image(cam_ours_img,  x2,   img2_rgb)); axes[1].set_title("ours-x2"); axes[1].axis('off')
#         axes[2].imshow(get_cam_image(cam_ours_diff, diff, diff_rgb)); axes[2].set_title("ours-diff"); axes[2].axis('off')

#         # mb
#         axes[3].imshow(get_cam_image(cam_mb_img, x1, img1_rgb)); axes[3].set_title("mb-x1"); axes[3].axis('off')
#         axes[4].imshow(get_cam_image(cam_mb_img, x2, img2_rgb)); axes[4].set_title("mb-x2"); axes[4].axis('off')

#         # bb
#         axes[5].imshow(get_cam_image(cam_bb_diff, diff, diff_rgb)); axes[5].set_title("bb-diff"); axes[5].axis('off')

#         plt.tight_layout()
#         save_path = f"gradcam_results/sample_{idx}.png"
#         plt.savefig(save_path, dpi=200)
#         plt.close()
#         print(f"Saved: {save_path}")
