import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from torchvision import transforms
# 정세림 쉽서움
class PairDataset(Dataset):
    
    def __init__(self, image_dir, pairs_file, transform=None):
        """
        Args:
            image_dir (str): 이미지가 저장된 디렉토리 경로
            pairs_file (str): 이미지 쌍 정보가 담긴 파일 경로
                            각 줄은 "image1_path,image2_path,label" 형식
                            (label: 1=positive pair, 0=negative pair)
            transform: 이미지 변환 함수
        """
        self.image_dir = image_dir
        self.transform = transform
        self.basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])
        
        # 쌍 정보를 한번에 로드
        data = np.loadtxt(pairs_file, delimiter='\t', dtype=str)
        
        # # glasses_pair가 2인 데이터만 필터링
        # valid_mask = np.isin(data[:,3].astype(np.int32), [0])
        print(f"Total pairs loaded: {len(data)}")
        # data = data[valid_mask]
        
        self.pairs = list(zip(data[:,0], data[:,1]))
        self.labels = data[:,2].astype(np.int32)
        self.glasses_pair = data[:,3].astype(np.int32)
        
        # 이미지 경로 미리 생성
        self.img1_paths = [p[0] for p in self.pairs]
        self.img2_paths = [p[1] for p in self.pairs]
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        # OpenCV 로드 (BGR -> RGB)
        img1 = cv2.imread(self.img1_paths[idx]);  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(self.img2_paths[idx]);  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # 항상 기본 전처리 먼저 (PIL→Resize→Tensor, [0,1] float)
        img1_t = self.basic_transform(img1)   # torch.Tensor [3,224,224]
        img2_t = self.basic_transform(img2)
        
        img1_ori = img1_t.clone()
        img2_ori = img2_t.clone()

        # 추가 transform 이 있으면 텐서 기준으로 적용 (주의: 여기선 PIL이 아니라 Tensor!)
        if self.transform is not None:
            img1_t = self.transform(img1_t)
            img2_t = self.transform(img2_t)

        glasses_label1 = 1 if 'glass' in self.img1_paths[idx] and 'no_glass' not in self.img1_paths[idx] else 0
        glasses_label2 = 1 if 'glass' in self.img2_paths[idx] and 'no_glass' not in self.img2_paths[idx] else 0
        

        return (
            img1_t,  # 변환본(모델 입력)
            img2_t,  # 변환본(모델 입력)
            torch.tensor(glasses_label1, dtype=torch.long),
            torch.tensor(glasses_label2, dtype=torch.long),
            torch.FloatTensor([self.labels[idx]]),
            torch.FloatTensor([self.glasses_pair[idx]]),
            img1_ori,   # 원본 대신 '기본전처리된' 텐서를 돌려도 CAM/시각화에 충분함
            img2_ori
        )
import os

class PeriodicularDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        주기적 특징 데이터셋 초기화
        Args:
            data_dir (str): D:/sampling_data_origin/train 경로
            transform: 이미지 전처리를 위한 변환 함수
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.person_to_label = {}
        
        # 사람별 레이블 매핑 생성
        for person_id in sorted(os.listdir(data_dir)):
            if person_id not in self.person_to_label:
                self.person_to_label[person_id] = len(self.person_to_label)
            
            person_dir = os.path.join(data_dir, person_id)
            if not os.path.isdir(person_dir):
                continue
                
            # 세션별 이미지 로드
            for session in os.listdir(person_dir):
                session_dir = os.path.join(person_dir, session)
                if not os.path.isdir(session_dir):
                    continue
                    
                for img_name in os.listdir(session_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(session_dir, img_name))
                        self.labels.append(self.person_to_label[person_id])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        데이터셋에서 하나의 샘플을 가져옴
        Args:
            idx (int): 샘플의 인덱스
        Returns:
            tuple: (이미지 텐서, 레이블)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # OpenCV로 이미지 로드 (BGR -> RGB 변환)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label 