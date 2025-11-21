import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from models.shvit.build import shvit_s2
import torchvision

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.device)
        
        if config.model_name == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT if config.pretrained else None)
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif config.model_name == "mobilenetv3":
            self.model = torchvision.models.mobilenet_v3_large(pretrained=config.pretrained)
            self.feature_dim = self.model.classifier[0].in_features # 1280? No, MobileNetV3 Large last conv is 960 usually, but classifier input is 960.
            # Wait, let's check MobileNetV3 structure.
            # torchvision mobilenet_v3_large:
            # features -> avgpool -> classifier
            # classifier is Sequential(Linear(960, 1280), Hardswish, Dropout, Linear(1280, num_classes))
            # If we replace classifier with Identity, the output is from avgpool, which is 960.
            # But wait, the code says: self.model.classifier = nn.Identity()
            # So the output of self.model(x) will be the output of avgpool (flattened).
            # The output channels of the last block of features is 960.
            self.feature_dim = 960 
            self.model.classifier = nn.Identity()
        elif config.model_name == "shvit_s2":
            # 1, 448
            self.model = shvit_s2()
            self.feature_dim = 512 # Based on metric_model.py logic? No, let's check shvit.py if possible, or assume 512 as per metric_model.py
            # In metric_model.py: feature_dim = 448 or 512?
            # It says: feature_dim = feature_extractor.model.head.in_features
            # I should check shvit.py or just set it dynamically.
            if hasattr(self.model, 'head'):
                 self.feature_dim = self.model.head.in_features
            else:
                 self.feature_dim = 512 # Fallback
            self.model.head = nn.Identity()

        self.model.to(self.device)
    
    def forward(self, x):
        features = self.model(x)
        return features