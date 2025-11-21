import torch
import torch.nn as nn
import torch.nn.functional as F

class RockSiN(nn.Module):
    def __init__(self, feature_extractor, domain_feature_extractor):
        super(RockSiN, self).__init__()
        self.feature_extractor = feature_extractor
        self.domain_feature_extractor = domain_feature_extractor
        
        # 모델 타입에 따라 feature_dim 결정
        if hasattr(feature_extractor, 'feature_dim'):
            feature_dim = feature_extractor.feature_dim
        else:
            # Fallback logic
            if hasattr(feature_extractor.model, 'fc'):
                if isinstance(feature_extractor.model.fc, nn.Identity):
                    feature_dim = 2048
                else:
                    feature_dim = feature_extractor.model.fc.in_features
            elif hasattr(feature_extractor.model, 'classifier'):
                if isinstance(feature_extractor.model.classifier, nn.Identity):
                    in_channels = feature_extractor.model.features[-1][0].in_channels
                    out_channels = feature_extractor.model.features[-1][0].out_channels
                    feature_dim = out_channels
                else:
                    feature_dim = feature_extractor.model.classifier[0].in_features
            else:
                feature_dim = 1280

        self.fc = nn.Sequential(
            nn.Linear(int(feature_dim) * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(feature_dim), 512),
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 1),  
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return features

    def biometric_from_diff(self, x1, x2):
        img_diff = x1 - x2
        diff_feat = self.domain_feature_extractor(img_diff)
        logit = self.fc2(diff_feat)
        prob  = torch.sigmoid(logit)
        return prob, logit
    
    def biometric_from_siamese(self,x1,x2):
        with torch.set_grad_enabled(self.fc2.training):
            feat1 = self.forward(x1)
            feat2 = self.forward(x2)
            diff = torch.abs(feat1 - feat2)
            logit = self.fc2(diff)
            prob  = torch.sigmoid(logit)
        return prob, logit
    def forward_pair(self, x1, x2,data_ori1,data_ori2):

        # 두 이미지의 차이 이미지를 모델에 통과시켜 특징 추출
        img_diff = data_ori1 - data_ori2
        diff_feat = self.domain_feature_extractor(img_diff)
        
        # 각 이미지에서 특징 추출
        feat1 = self.forward(x1)
        feat2 = self.forward(x2)

        # 두 특징 벡터의 차이 계산
        diff = torch.abs(feat1 - feat2)

        feature = torch.cat([diff, diff_feat], dim=1)

        # FC 레이어를 통과시켜 유사도 점수 출력 (0~1 사이)
        similarity_beforesig = self.fc(feature)
        similarity = torch.sigmoid(similarity_beforesig)
        return similarity, similarity_beforesig

class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss (이진 교차 엔트로피 손실 함수)
    """
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, similarity, label):
        """
        Args:
            similarity: 모델이 예측한 유사도 (0~1 사이)
            label: 1 (같은 클래스, positive pair), 0 (다른 클래스, negative pair)
        """
        return self.bce_loss(similarity, label.float()) 