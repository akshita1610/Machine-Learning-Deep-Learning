# PyTorch Image Classifier with Transfer Learning

A production-ready image classification pipeline built with PyTorch, demonstrating both custom CNN architectures and transfer learning techniques. This project classifies images from the CIFAR-10 dataset (10 classes) with comprehensive training, evaluation, and visualization tools.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [License](#-license)

## 🌟 Features

- **Dual Model Architecture**: Custom CNN from scratch + pretrained ResNet18 with transfer learning
- **Complete Training Pipeline**: Batch processing, loss tracking, validation monitoring, model checkpointing
- **Data Augmentation**: Rotation, flip, crop, and color jittering for improved generalization
- **Comprehensive Evaluation**: Test accuracy, confusion matrix, precision/recall/F1 metrics
- **Visual Analytics**: Training curves, loss plots, sample predictions grid
- **Command-Line Interface**: Easy-to-use CLI for training and evaluation with configurable hyperparameters
- **Modular Codebase**: Clean separation of concerns across data loading, model definition, training, and evaluation

## 🛠 Tech Stack

- **Python 3.8+**
- **PyTorch 2.0+** - Deep learning framework
- **TorchVision** - Pretrained models and datasets
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization
- **scikit-learn** - Metrics and evaluation
- **tqdm** - Progress bars

## 📊 Dataset

This project uses the **CIFAR-10** dataset:
- 60,000 32×32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images + 10,000 test images
- Automatically downloaded on first run

## 📁 Project Structure

```
pytorch-image-classifier/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading and preprocessing
│   ├── model.py             # Model architectures (Custom CNN + ResNet)
│   ├── train.py             # Training script with CLI
│   ├── evaluate.py          # Evaluation and visualization
│   └── utils.py             # Helper functions
├── notebooks/               # Jupyter notebooks (optional)
│   └── experiments.ipynb
├── data/                    # Dataset storage (auto-downloaded)
└── outputs/                 # Training outputs
    ├── checkpoints/         # Saved model weights
    ├── training_curves.png  # Loss and accuracy plots
    ├── confusion_matrix.png # Confusion matrix visualization
    ├── sample_predictions.png # Sample predictions grid
    └── test_results.txt     # Evaluation metrics
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pytorch-image-classifier.git
   cd pytorch-image-classifier
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Training

Train a custom CNN model:
```bash
python src/train.py --model custom --epochs 30 --batch_size 64 --learning_rate 0.001
```

Train with pretrained ResNet18 (transfer learning):
```bash
python src/train.py --model resnet --pretrained --epochs 20 --batch_size 32
```

Resume training from checkpoint:
```bash
python src/train.py --model custom --resume outputs/checkpoints/latest_model.pth --epochs 50
```

#### Training Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture: `custom` or `resnet` | `custom` |
| `--pretrained` | Use pretrained weights (for ResNet) | `False` |
| `--epochs` | Number of training epochs | `30` |
| `--batch_size` | Batch size | `64` |
| `--learning_rate` | Learning rate | `0.001` |
| `--data_dir` | Dataset directory | `./data` |
| `--output_dir` | Output directory | `./outputs` |
| `--resume` | Path to checkpoint to resume from | `None` |

### Evaluation

Evaluate the trained model:
```bash
python src/evaluate.py --model custom --model_path outputs/checkpoints/best_model.pth
```

This will:
- Calculate test accuracy
- Generate confusion matrix
- Show classification report (precision, recall, F1-score)
- Create sample predictions visualization

## 📈 Results

### CustomCNN Performance
- **Test Accuracy**: ~75-80% (after 30 epochs)
- **Parameters**: ~500K trainable parameters
- **Training Time**: ~10-15 minutes on GPU, ~30-45 minutes on CPU

### ResNet18 Transfer Learning Performance
- **Test Accuracy**: ~85-90% (after 20 epochs)
- **Parameters**: ~11M total, ~500K fine-tuned
- **Training Time**: ~15-20 minutes on GPU

### Example Outputs

After training, you'll find:
- `training_curves.png` - Training/validation loss and accuracy curves
- `confusion_matrix.png` - Normalized confusion matrix
- `sample_predictions.png` - Grid of test images with predictions
- `test_results.txt` - Detailed classification report

## 🔮 Future Improvements

### Model Enhancements
- [ ] Implement additional architectures (VGG, EfficientNet, Vision Transformer)
- [ ] Add learning rate warmup and cosine annealing schedules
- [ ] Implement mixup/cutmix data augmentation
- [ ] Add model ensemble for better accuracy

### Features
- [ ] Create interactive web UI for predictions
- [ ] Add support for custom datasets
- [ ] Implement model pruning and quantization
- [ ] Add TensorBoard logging integration
- [ ] Create Docker container for easy deployment

### Code Quality
- [ ] Add unit tests with pytest
- [ ] Set up CI/CD with GitHub Actions
- [ ] Add code linting (flake8, black)
- [ ] Create detailed documentation with Sphinx

## 🎓 Learning Resources

This project is perfect for:
- Computer Science students learning deep learning
- Portfolio projects for job applications
- Understanding PyTorch fundamentals
- Learning transfer learning techniques

## 📝 Tips for GitHub Portfolio

1. **Add a screenshot** of your results to the README
2. **Create releases** with pretrained models
3. **Write a blog post** explaining your implementation
4. **Add badges** for build status, code coverage, etc.
5. **Include a CONTRIBUTING.md** for open-source collaboration

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky
- PyTorch team for the amazing framework
- ResNet paper: "Deep Residual Learning for Image Recognition"

---

*Built with ❤️ for learning and sharing knowledge.*
