# RockSiN: Robust Periocular Recognition

**RockSiN** is a deep learning-based periocular recognition system designed for robust verification, particularly effective with Near-Infrared (NIR) images. It leverages a Siamese network architecture combined with difference feature learning to achieve high accuracy in matching periocular image pairs.

## 🚀 Features

- **Siamese Architecture**: Utilizes a dual-branch network to extract features from image pairs.
- **Difference Feature Learning**: Explicitly learns from the difference between feature vectors to improve discrimination.
- **Flexible Backbones**: Supports various backbone networks including:
  - **MobileNetV3**: Lightweight and fast, suitable for edge devices.
  - **ResNet18**: A robust standard baseline.
  - **SHViT**: Single-Head Vision Transformer for capturing global context.
- **Easy Configuration**: Fully configurable via command-line arguments.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RockSiN.git
   cd RockSiN
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision numpy opencv-python tqdm scikit-learn matplotlib
   ```

## 🏃‍♂️ Usage

### Training

To train the model, use `main.py`. You can specify the backbone, batch size, epochs, and other hyperparameters.

```bash
# Train with MobileNetV3 (default)
python main.py --exp_name my_experiment --epochs 20

# Train with ResNet18
python main.py --model_name resnet18 --batch_size 64 --lr 0.001
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | Backbone model (`mobilenetv3`, `resnet18`, `shvit_s2`) | `mobilenetv3` |
| `--data_dir` | Path to dataset directory | `./periocular_data` |
| `--epochs` | Number of training epochs | 10 |
| `--batch_size` | Batch size | 32 |
| `--lr` | Learning rate | 0.0001 |
| `--fold` | Fold number for cross-validation | 1 |

## 📂 Project Structure

```
RockSiN/
├── data/           # Dataset loading and processing (PairDataset)
├── models/         # Model definitions (RockSiN, FeatureExtractors)
├── training/       # Training loops and evaluation logic
├── tools/          # Utility scripts (GradCAM, visualization, etc.)
├── notebooks/      # Jupyter notebooks for analysis
└── main.py         # Entry point for training
```

## 📊 Demo

We provide a web-based demo to test the model with your own images. 
*(Instructions for running the demo will be added here)*

## 📝 License

This project is licensed under the MIT License.
