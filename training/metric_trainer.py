import torch
import matplotlib
matplotlib.use("Agg") 
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class RockSiNTrainer:
    
    def __init__(self, model, criterion, optimizer, device, distance_threshold=0.3):
        """
        RockSiN 학습 트레이너 초기화
        Args:
            model: 학습할 모델
            criterion: 손실 함수
            optimizer: 옵티마이저
            device: 학습에 사용할 디바이스 (cuda/cpu)
            distance_threshold: 양성/음성 쌍 판단을 위한 거리 임계값
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.distance_threshold = distance_threshold
        self.domain_criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        """
        대조 학습을 위한 한 에폭 학습
        Args:
            train_loader: 학습 데이터 로더 (쌍으로 된 데이터 제공)
        Returns:
            float: 평균 학습 손실
        """
        self.model.train()  # 전체 모델 학습 모드
        total_loss = 0
        loss_list = []
        
        pbar = tqdm(train_loader)
        for batch_idx, (data1, data2, glasses_label1, glasses_label2, labels, glasses_pair,data_ori1,data_ori2) in enumerate(pbar):
            data1, data2, labels, glasses_pair,data_ori1,data_ori2 = data1.to(self.device), data2.to(self.device), labels.to(self.device), glasses_pair.to(self.device),data_ori1.to(self.device),data_ori2.to(self.device)
            glasses_label1, glasses_label2 = glasses_label1.to(self.device), glasses_label2.to(self.device)

            self.optimizer.zero_grad()
            
            # 시암즈 네트워크만한거 
            prob,logit = self.model.forward_pair(data1, data2, data_ori1, data_ori2)
            # BCE 손실 계산
            loss_bce = self.criterion(prob, labels.float())
            
            loss = loss_bce
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            loss_list.append(loss.item())
            total_loss += loss.item()
            
            # 현재까지의 평균 손실을 진행 막대에 표시
            pbar.set_description(f'Average Loss: {sum(loss_list)/len(loss_list):.4f}')
                    
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, dataloader, epoch, model_name="default"):
        self.model.eval()

        # 저장 디렉토리 생성
        save_dir = os.path.join("results", model_name, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        total_loss = 0
        correct = 0
        total = 0

        all_similarities_raw = []
        all_labels = []
        all_glasses_pair = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            img1, img2, glasses_label1, glasses_label2, label, glasses_pair,ori_img1,ori_img2 = [b.to(self.device) for b in batch]

            with torch.no_grad():
                similarity, before = self.model.forward_pair(img1, img2, ori_img1, ori_img2)
                loss = self.criterion(similarity, label)
            
            total_loss += loss.item()
            pred = (similarity > 0.5).float()
            correct += (pred.squeeze() == label).sum().item()
            total += label.size(0)

            all_similarities_raw.extend(before.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_glasses_pair.extend(glasses_pair.cpu().numpy())

        similarities = np.array(all_similarities_raw)  # sigmoid 이전값
        labels = np.array(all_labels)
        glasses_pairs = np.array(all_glasses_pair)

        # 히스토그램 저장 함수
        def save_histogram(title, filename, masks):
            plt.figure(figsize=(8, 6))
            for mask, label in masks:
                plt.hist(similarities[mask], bins=50, alpha=0.5, label=label, density=True)
            plt.xlabel("Raw Similarity Score (before sigmoid)")
            plt.ylabel("Density")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()

        # glasses_pair별 분포 저장
        for pair_val in [0, 1, 2]:
            mask_genuine = (labels == 1) & (glasses_pairs == pair_val)
            mask_impostor = (labels == 0) & (glasses_pairs == pair_val)
            save_histogram(
                f"Similarity Distribution (Pair {pair_val}) - Epoch {epoch}",
                f"hist_pair{pair_val}.png",
                [(mask_genuine, "Genuine"), (mask_impostor, "Impostor")]
            )

        # 전체 genuine/impostor
        save_histogram(
            f"Genuine Distribution Comparison - Epoch {epoch}",
            f"genuine_compare.png",
            [((labels == 1) & (glasses_pairs == val), f"Pair {val}") for val in [0, 1, 2]]
        )

        save_histogram(
            f"Impostor Distribution Comparison - Epoch {epoch}",
            f"impostor_compare.png",
            [((labels == 0) & (glasses_pairs == val), f"Pair {val}") for val in [0, 1, 2]]
        )

        # 전체
        save_histogram(
            f"Overall Score Distribution - Epoch {epoch}",
            f"overall_compare.png",
            [(glasses_pairs == val, f"Pair {val}") for val in [0, 1, 2]]
        )

        # ROC, EER 등 계산
        fpr, tpr, thresholds = roc_curve(labels, similarities, pos_label=1)
        roc_auc = auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        zero_ffr = fpr[np.argmin(np.abs(fnr))]
        zero_far = fnr[np.argmin(np.abs(fpr))]

        # suspicious impostor 쌍 저장
        impostor_mask = (labels == 0)
        suspicious_indices = np.where(impostor_mask & (similarities > 0))[0]
        suspicious_file = os.path.join(save_dir, f"suspicious_pairs_epoch_{epoch}.txt")
        with open(suspicious_file, "w") as f:
            for idx in suspicious_indices:
                try:
                    path1 = dataloader.dataset.img1_paths[idx]
                    path2 = dataloader.dataset.img2_paths[idx]
                    sim_score = similarities[idx]
                    f.write(f"{path1}\t{path2}\t{sim_score:.4f}\n")
                except:
                    pass  # img1_paths 없을 수 있음

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total,
            "auc": roc_auc,
            "eer": eer,
            "zero_ffr": zero_ffr,
            "zero_far": zero_far
        }