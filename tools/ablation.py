# run.py  (siam-only, encoder freeze + head 학습)
import warnings

# 모든 경고 무시
warnings.filterwarnings("ignore")
import os, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config.config import ModelConfig
from models.feature_extractor import FeatureExtractor
from data.dataset import PairDataset
from models.metric_model import ContrastiveModelAblation
from training.metric_trainer import ContrastiveTrainerAblation

# -------------------- utils --------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_encoders_from_ckpt(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    # case 1: flat state_dict with prefixes
    if isinstance(sd, dict) and any(k.startswith('feature_extractor.') for k in sd.keys()):
        fe_sd = {k.replace('feature_extractor.', ''): v
                 for k, v in sd.items() if k.startswith('feature_extractor.')}
        de_sd = {k.replace('domain_feature_extractor.', ''): v
                 for k, v in sd.items() if k.startswith('domain_feature_extractor.')}
        model.feature_extractor.load_state_dict(fe_sd, strict=False)
        model.domain_feature_extractor.load_state_dict(de_sd, strict=False)
        return
    # case 2: wrapped dict
    if 'feature_extractor' in sd and isinstance(sd['feature_extractor'], dict):
        model.feature_extractor.load_state_dict(sd['feature_extractor'], strict=False)
    if 'domain_feature_extractor' in sd and isinstance(sd['domain_feature_extractor'], dict):
        model.domain_feature_extractor.load_state_dict(sd['domain_feature_extractor'], strict=False)

# -------------------- dataloaders --------------------
def make_loaders(image_dir, fold, batch_size, num_workers=4):
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_ds = PairDataset(
        image_dir=image_dir,
        pairs_file=os.path.join(image_dir, f"fold{fold}_pairs_train.txt"),
        transform=train_tf
    )
    val_ds = PairDataset(
        image_dir=image_dir,
        pairs_file=os.path.join(image_dir, f"fold{fold}_pairs_valid.txt"),
        transform=val_tf
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=num_workers,
                              prefetch_factor=4, persistent_workers=(num_workers>0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=num_workers,
                            prefetch_factor=4, persistent_workers=(num_workers>0))
    return train_loader, val_loader
import sys
# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="./periocular_data")
    parser.add_argument("--ckpt", type=str, default="./best_weight/ours.pth")
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--backbone", type=str, default="mobilenetv3")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="ablation_siam")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{args.model_name}_train_log.txt")
    # loaders
    train_loader, val_loader = make_loaders(
        image_dir=args.image_dir, fold=args.fold,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # encoders
    model_cfg = ModelConfig(model_name=args.backbone, pretrained=args.pretrained, device=device)
    feat_extractor = FeatureExtractor(model_cfg)
    dom_extractor  = FeatureExtractor(model_cfg)

    # model (siam only)
    model = ContrastiveModelAblation(
        feature_extractor=feat_extractor,
        domain_feature_extractor=dom_extractor,
        mode="siam"
    ).to(device)

    # load encoders only + freeze + set mode
    _load_encoders_from_ckpt(model, args.ckpt, device)
    model.freeze_encoders()
    model.set_mode("siam")

    # trainer (head-only optimization inside)
    trainer = ContrastiveTrainerAblation(model, device, lr=args.lr, mode="siam")

    # loop
    for epoch in range(1, args.epochs + 1):
        sys.stdout = open(log_path, 'w', encoding='utf-8')
        tr_loss = trainer.train_epoch(train_loader)
        
        metrics = trainer.evaluate(val_loader, epoch, model_name=args.model_name)
        print(f"[siam] fold {args.fold} | epoch {epoch} | "
              f"loss {tr_loss:.4f} | AUC {metrics['auc']:.4f} | EER {metrics['eer']:.4f}")
        
    sys.stdout.close()

if __name__ == "__main__":
    main()
