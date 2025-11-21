import cv2
import numpy as np
import torch
import os
from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor
from model import hg2
from torch.nn.parallel import DataParallel

# 좌표 추출 함수
def getCoordFromHeatmap(heatmap):
    max_value = heatmap.max()
    max_value_index = np.where(heatmap == max_value)
    return max_value_index[0][0], max_value_index[1][0]

# EAR
def calculate_ear(eyelid_pred):
    horizontal_dist = np.linalg.norm(eyelid_pred[11] - eyelid_pred[6])
    vertical_dist1 = np.linalg.norm(eyelid_pred[2] - eyelid_pred[10])
    vertical_dist2 = np.linalg.norm(eyelid_pred[4] - eyelid_pred[8])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

# RANSAC 기반 원 피팅 함수
def fit_circle_ransac(points):
    """RANSAC을 사용하여 원 피팅"""
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    model_x = RANSACRegressor().fit(x, y)
    model_y = RANSACRegressor().fit(y.reshape(-1, 1), x)

    x0 = model_x.predict(np.array([[np.median(x)]]))[0]
    y0 = model_y.predict(np.array([[np.median(y)]]))[0]

    r = np.mean(np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2))
    return x0, y0, r

# 동그란 정도(원형도) 계산
def calculate_roundness(pupil_pred):
    """동공 좌표를 기반으로 원형도를 계산"""
    if len(pupil_pred) != 8:
        return 0  # 좌표 부족 시 최악의 원형도로 처리

    x0, y0, r = fit_circle_ransac(pupil_pred)
    if r < 5:  # 동공 반지름이 너무 작으면 예외 처리
        return 0

    distances = np.sqrt((pupil_pred[:, 0] - x0) ** 2 + (pupil_pred[:, 1] - y0) ** 2)
    max_deviation = np.max(np.abs(distances - r)) / r  # 평균 반지름 대비 최대 편차율
    return max_deviation

# 동공 좌표 중복 확인
def has_duplicate_points(points):
    """동공 좌표 배열에서 중복된 점이 있는지 확인"""
    unique_points = np.unique(points, axis=0)
    unique_points = np.unique(unique_points, axis = 1)
    return len(unique_points) < len(points)  # 중복된 점이 하나라도 있으면 True

# 유효한 동공 좌표인지 확인하는 함수
def is_valid_pupil(pupil_pred, min_value=5):
    if len(pupil_pred) != 8:
        return False
    if np.isnan(pupil_pred).any():
        return False
    if np.any(pupil_pred < min_value):  # 좌표값이 너무 작으면 무효
        return False
    return True

# 모델 설정
model = hg2().cuda()
model = DataParallel(model).to('cuda')
model_weight_dir = './checkpoint.pth.tar'
model.load_state_dict(torch.load(model_weight_dir)['state_dict'])
model.eval()

# 기존 영상이 있는 폴더
base_video_folder = r'D:\iris_data'
# 새로운 저장 폴더
output_base_folder = r'D:\iris_processed'

# 모든 하위 폴더의 영상을 찾음
for root, _, files in os.walk(base_video_folder):
    for file in files:
        if file.endswith('.mp4'):
            video_path = os.path.join(root, file)
            print(f"Processing video: {video_path}")

            # 상대 경로 추출 (기존 경로 구조 유지)
            relative_path = os.path.relpath(root, base_video_folder)  
            save_folder = os.path.join(output_base_folder, relative_path, os.path.splitext(file)[0])  
            os.makedirs(save_folder, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if file.startswith('eye0'):
                    frame = cv2.flip(frame, 0)

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (256, 256)).reshape(1, 256, 256) / 255.0
                img = torch.from_numpy(img).float().unsqueeze(0).cuda()
                
                output = model(img)
                
                pred = output[0].cpu().detach().numpy()
                pred_coord = np.array([getCoordFromHeatmap(pred[0][i]) for i in range(28)])
                
                for i in range(28):
                    pred_coord[i] = [pred_coord[i][1], pred_coord[i][0]]
                
                h, w = frame.shape[:2]
                ratio_x = w / 64
                ratio_y = h / 64
                
                eyelid_pred = pred_coord[0:12].copy()
                iris_pred = pred_coord[12:20].copy()
                pupil_pred = pred_coord[20:].copy()
                
                for i in range(len(eyelid_pred)):
                    eyelid_pred[i][0] *= ratio_x
                    eyelid_pred[i][1] *= ratio_y
                    
                for i in range(len(iris_pred)):
                    iris_pred[i][0] *= ratio_x
                    iris_pred[i][1] *= ratio_y
                    
                for i in range(len(pupil_pred)):
                    pupil_pred[i][0] *= ratio_x
                    pupil_pred[i][1] *= ratio_y
                
                # 눈이 떠 있는지 판단
                if is_valid_pupil(pupil_pred) and not has_duplicate_points(pupil_pred):
                    ear_value = calculate_ear(eyelid_pred)
                    roundness_score = calculate_roundness(pupil_pred)
                    eye_state = "Open" if ear_value > 0.25 and roundness_score > 0.075 else "Closed"
                else:
                    eye_state = "Closed"  # 동공 좌표가 중복되거나 이상하면 강제로 닫힌 눈으로 처리

                if eye_state == "Open":
                    save_path = os.path.join(save_folder, f"frame_{frame_idx:05d}.jpg")
                    cv2.imwrite(save_path, frame)
                    print(f"Saved: {save_path}")

                frame_idx += 1

            cap.release()
