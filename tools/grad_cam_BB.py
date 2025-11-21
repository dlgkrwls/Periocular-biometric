import torch
from torchvision import transforms
from data.dataset import PairDataset
from config.config import ModelConfig
from models.feature_extractor import FeatureExtractor
from models.metric_model import ContrastiveModel_only_BB
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 모델 로드
def load_model(weight_path):
    model_config = ModelConfig(model_name="mobilenetv3", pretrained=False, device=device)
    feat_extractor = FeatureExtractor(model_config)
    model = ContrastiveModel_only_BB(feat_extractor).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.eval()
    return model
def load_model_mb(weight_path):
    model_config = ModelConfig(model_name="mobilenetv3", pretrained=False, device=device)
    feat_extractor = FeatureExtractor(model_config)
    model = ContrastiveModel_only_BB(feat_extractor).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.eval()
    return model

# 2) 데이터 준비
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = PairDataset(
    image_dir="./periocular_data",
    pairs_file="./periocular_data/fold1_pairs_valid.txt",
    transform=transform
)

# 3) 모델 준비
model_bb = load_model("./best_weight/bb.pth")
model_mb = load_model("./best_weight/mb.pth")  
model_ours = load_model("./best_weight/ours.pth")
target_layers = [model.feature_extractor.model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# 4) 4개의 샘플에 대해 시각화
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

sample_indices = [0, 10, 20, 30]  # 원하는 인덱스 지정
for ax, idx in zip(axes.flatten(), sample_indices):
    img1_t, img2_t, *_ , img1_ori, img2_ori = dataset[idx]
    x1 = img1_ori.unsqueeze(0).to(device)
    x2 = img2_ori.unsqueeze(0).to(device)

    # 차영상 생성
    img_diff = (x1 - x2).to(device)

    # Grad-CAM 실행
    grayscale_cam = cam(input_tensor=img_diff)[0]
    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))

    # 차영상 시각화를 위해 0~1 정규화
    img_diff_np = img_diff.squeeze().permute(1, 2, 0).cpu().numpy()
    img_diff_np = (img_diff_np - img_diff_np.min()) / (img_diff_np.max() - img_diff_np.min())

    # CAM 오버레이
    cam_image = show_cam_on_image(img_diff_np, grayscale_cam, use_rgb=True)

    ax.imshow(cam_image)
    ax.set_title(f"Sample {idx}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("gradcam_results/gradcam_diff_bbonly_4samples.png")
plt.show()

print("Grad-CAM 4개 샘플 시각화 저장 완료.")
