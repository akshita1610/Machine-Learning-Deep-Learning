# MNIST Digit Recognition using Convolutional Neural Network (CNN)

A comprehensive implementation of handwritten digit recognition using a Convolutional Neural Network (CNN) on the MNIST dataset. This project demonstrates fundamental deep learning concepts and is designed as a second-year computer science coursework project.

## 🎯 Project Overview

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels in size. This project builds a CNN model that learns to classify these digits with high accuracy, demonstrating the power of deep learning for image recognition tasks.

### Why CNNs for Image Recognition?

Convolutional Neural Networks are particularly effective for image recognition because:

- **Local Connectivity**: CNNs use local receptive fields to capture spatial patterns
- **Weight Sharing**: Convolutional layers share weights across spatial locations, reducing parameters
- **Hierarchical Features**: Multiple layers learn increasingly complex features (edges → shapes → objects)
- **Translation Invariance**: Pooling layers provide robustness to small translations
- **Efficient Learning**: Fewer parameters compared to fully connected networks for similar performance

## 📁 Project Structure

```
MNIST Digit Recognition (CNN)/
├── mnist_cnn.py              # Main implementation file
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── models/                  # Directory for saved models
│   └── mnist_cnn_model.h5   # Trained model weights
├── images/                  # Directory for visualization outputs
│   ├── training_history.png
│   ├── sample_predictions.png
│   └── confusion_matrix.png
└── notebooks/               # Directory for Jupyter notebooks (optional)
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   # Navigate to your projects directory
   cd "C:\Users\akshi\OneDrive\Desktop\Projects"
   
   # The project should already be in: "MNIST Digit Recognition (CNN)"
   cd "MNIST Digit Recognition (CNN)"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv mnist_env
   
   # Activate virtual environment
   # Windows:
   mnist_env\Scripts\activate
   # macOS/Linux:
   source mnist_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Complete Pipeline

Execute the main script to run the complete training and evaluation pipeline:

```bash
python mnist_cnn.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Build the CNN model
3. Train the model for 15 epochs
4. Evaluate performance on the test set
5. Generate visualizations and save the trained model

### Using the MNISTDigitRecognizer Class

You can also use the class programmatically for more control:

```python
from mnist_cnn import MNISTDigitRecognizer

# Initialize the recognizer
recognizer = MNISTDigitRecognizer()

# Load and preprocess data
recognizer.load_and_preprocess_data(validation_split=0.1)

# Build and train the model
recognizer.build_model()
recognizer.train_model(epochs=10, batch_size=128)

# Evaluate the model
test_loss, test_accuracy = recognizer.evaluate_model()

# Generate visualizations
recognizer.plot_training_history()
recognizer.plot_sample_predictions()
recognizer.plot_confusion_matrix()

# Save the model
recognizer.save_model()
```

## 🏗️ Model Architecture

The CNN architecture consists of:

### Convolutional Blocks
- **Block 1**: 32 filters → BatchNorm → 32 filters → BatchNorm → MaxPooling → Dropout
- **Block 2**: 64 filters → BatchNorm → 64 filters → BatchNorm → MaxPooling → Dropout

### Fully Connected Layers
- **Flatten Layer**: Converts 2D feature maps to 1D vector
- **Dense Layer**: 512 neurons with ReLU activation and BatchNorm
- **Dropout Layer**: 50% dropout for regularization
- **Output Layer**: 10 neurons with softmax activation

### Key Features
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout Regularization**: Prevents overfitting
- **Adam Optimizer**: Adaptive learning rate optimization
- **Categorical Crossentropy**: Suitable for multi-class classification

## 📊 Performance Metrics

### Expected Results
- **Training Accuracy**: >99%
- **Validation Accuracy**: >98%
- **Test Accuracy**: 98-99%
- **Training Time**: ~5-10 minutes on CPU, ~1-2 minutes on GPU

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Detailed error analysis per digit
- **Classification Report**: Precision, recall, and F1-score per class

## 📈 Visualizations

The project generates three main visualizations:

1. **Training History**: Plots of accuracy and loss over epochs
2. **Sample Predictions**: Visual comparison of true vs. predicted labels
3. **Confusion Matrix**: Heatmap showing classification patterns

All visualizations are saved in the `images/` directory as PNG files.

## 🔧 Customization

### Hyperparameter Tuning

You can easily modify training parameters:

```python
# Train with different parameters
recognizer.train_model(
    epochs=20,           # Increase for better performance
    batch_size=64,      # Smaller batch size for more updates
)
```

### Model Architecture

The model can be modified in the `build_model()` method:

```python
# Add more convolutional layers
layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
# Change dropout rates
layers.Dropout(0.3),  # Instead of 0.25
# Modify dense layer size
layers.Dense(1024, activation='relu'),  # Instead of 512
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size if you get GPU memory errors
2. **Slow Training**: Use a GPU or reduce the number of epochs
3. **Poor Performance**: Check data preprocessing and model architecture

### Solutions

```python
# For memory issues
recognizer.train_model(batch_size=32)  # Reduce from 128

# For slow training
recognizer.train_model(epochs=5)  # Reduce from 15

# For reproducible results
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
```

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.13.0 | Deep learning framework |
| numpy | 1.24.3 | Numerical computations |
| matplotlib | 3.7.1 | Plotting and visualization |
| seaborn | 0.12.2 | Statistical visualization |
| scikit-learn | 1.3.0 | Machine learning utilities |

## 🎓 Learning Objectives

This project helps students understand:

- **Deep Learning Fundamentals**: CNN architecture, backpropagation, optimization
- **Computer Vision**: Image preprocessing, feature extraction, classification
- **Model Evaluation**: Metrics, validation, overfitting prevention
- **Software Engineering**: Code organization, documentation, reproducibility

## 📖 Further Reading

- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by François Chollet
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - Interactive CNN visualization

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different architectures
- Try different hyperparameters
- Add data augmentation techniques
- Implement other evaluation metrics

## 📄 License

This project is for educational purposes. Feel free to use and modify for learning.

## 👨‍💻 Author

Second-year Computer Science Student  
March 2026

---

**Note**: This implementation is designed for educational purposes and demonstrates best practices for a second-year CS level project while maintaining clean, readable, and well-documented code.
