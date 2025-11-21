import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import argparse
import os
import sys
import random
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from types import SimpleNamespace
from models.feature_extractor import FeatureExtractor
from models.metric_model import RockSiN, BCELoss
from training.metric_trainer import RockSiNTrainer
from data.dataset import PairDataset

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_rocksin(args, train_loader, val_loader=None):
    device = torch.device(args.device)
    
    # 모델 설정
    model_config = SimpleNamespace(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )

    feature_extractor = FeatureExtractor(model_config)
    domain_feature_extractor = FeatureExtractor(model_config)
   
    model = RockSiN(feature_extractor=feature_extractor, domain_feature_extractor=domain_feature_extractor)
    model = model.to(device)
    
    print(f"Model: {args.model_name}, Device: {device}")
    print("TYPE:", type(model))

    # 손실 함수 및 옵티마이저
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    trainer = RockSiNTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        distance_threshold=args.distance_threshold
    )
  
    # 학습 루프
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}')
        
        # 모델 저장
        save_path = os.path.join(args.output_dir, args.exp_name, f'model_epoch_{epoch+1}.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(trainer.model.state_dict(), save_path)
        
        # 검증
        if val_loader:
            val_metrics = trainer.evaluate(val_loader, epoch, args.exp_name)
            print(f'Validation Results:')
            print(f'  Loss: {val_metrics["loss"]:.4f}')
            print(f'  Accuracy: {val_metrics["accuracy"]:.4f}')
            print(f'  AUC: {val_metrics["auc"]:.4f}')
            print(f'  EER: {val_metrics["eer"]:.4f}')

    return trainer

def main():
    parser = argparse.ArgumentParser(description="RockSiN Training Script")
    
    # Experiment Settings
    parser.add_argument('--exp_name', type=str, default='default_experiment', help='Experiment name for logging and saving weights')
    parser.add_argument('--output_dir', type=str, default='./weights', help='Directory to save model weights')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    
    # Data Settings
    parser.add_argument('--data_dir', type=str, default='C:/Users/user/Documents/GitHub/Periocular_Biometric-glasses/periocular_data', help='Root directory of the dataset')
    parser.add_argument('--fold', type=int, default=1, help='Fold number for cross-validation')
    
    # Model Settings
    parser.add_argument('--model_name', type=str, default='mobilenetv3', choices=['mobilenetv3', 'resnet18', 'shvit_s2'], help='Backbone model name')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained backbone')
    
    # Training Settings
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--distance_threshold', type=float, default=0.5, help='Distance threshold for prediction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    
    # Setup
    seed_everything(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Logging to file
    log_path = os.path.join(args.log_dir, f"{args.exp_name}_fold{args.fold}.log")
    sys.stdout = open(log_path, 'w', encoding='utf-8')
    print(f"Arguments: {args}")

    # Data Loaders
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = PairDataset(
        image_dir=args.data_dir,
        pairs_file=os.path.join(args.data_dir, f"fold{args.fold}_pairs_train.txt"),
        transform=train_transform
    )
    
    val_dataset = PairDataset(
        image_dir=args.data_dir,
        pairs_file=os.path.join(args.data_dir, f"fold{args.fold}_pairs_valid.txt"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=4, 
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4, 
        persistent_workers=True
    )

    # Start Training
    train_rocksin(args, train_loader, val_loader)
    
    sys.stdout.close()

if __name__ == "__main__":
    main()



