"""
MNIST Digit Recognition using Convolutional Neural Network (CNN)

This module implements a CNN for handwritten digit recognition using the MNIST dataset.
The project demonstrates fundamental deep learning concepts including:
- Data preprocessing and normalization
- CNN architecture with convolutional, pooling, and dense layers
- Model training and evaluation
- Visualization of results

Author: Second-year CS Student
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

class MNISTDigitRecognizer:
    """
    A class for building, training, and evaluating a CNN model for MNIST digit recognition.
    """
    
    def __init__(self):
        """Initialize the MNIST Digit Recognizer."""
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        
    def load_and_preprocess_data(self, validation_split=0.1):
        """
        Load and preprocess the MNIST dataset.
        
        Args:
            validation_split (float): Fraction of training data to use for validation
        """
        print("Loading MNIST dataset...")
        
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize pixel values to range [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data to include channel dimension (28x28x1)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # Split training data into training and validation sets
        num_val = int(len(x_train) * validation_split)
        self.x_val = x_train[:num_val]
        self.y_val = y_train[:num_val]
        self.x_train = x_train[num_val:]
        self.y_train = y_train[num_val:]
        
        # Convert labels to categorical one-hot encoding
        self.y_train = to_categorical(self.y_train, 10)
        self.y_val = to_categorical(self.y_val, 10)
        self.y_test = to_categorical(y_test, 10)
        self.x_test = x_test
        
        print(f"Training set: {self.x_train.shape[0]} samples")
        print(f"Validation set: {self.x_val.shape[0]} samples")
        print(f"Test set: {self.x_test.shape[0]} samples")
        
    def build_model(self):
        """
        Build the CNN model architecture.
        
        Returns:
            tensorflow.keras.Model: Compiled CNN model
        """
        print("Building CNN model...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
                         padding='same', name='conv1'),
            layers.BatchNormalization(name='bn1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
            layers.BatchNormalization(name='bn2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4'),
            layers.BatchNormalization(name='bn4'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Fully Connected Layers
            layers.Flatten(name='flatten'),
            layers.Dense(512, activation='relu', name='fc1'),
            layers.BatchNormalization(name='bn5'),
            layers.Dropout(0.5, name='dropout3'),
            layers.Dense(10, activation='softmax', name='output')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, epochs=15, batch_size=128):
        """
        Train the CNN model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        print(f"Training model for {epochs} epochs...")
        
        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("Evaluating model on test set...")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def predict_samples(self, num_samples=10):
        """
        Make predictions on sample test images.
        
        Args:
            num_samples (int): Number of samples to predict
        
        Returns:
            tuple: (true_labels, predicted_labels, sample_images)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Select random samples from test set
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        sample_images = self.x_test[indices]
        true_labels = np.argmax(self.y_test[indices], axis=1)
        
        # Make predictions
        predictions = self.model.predict(sample_images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        return true_labels, predicted_labels, sample_images
    
    def plot_training_history(self, save_path=None):
        """
        Plot training and validation accuracy/loss curves.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_sample_predictions(self, num_samples=10, save_path=None):
        """
        Plot sample predictions with true labels.
        
        Args:
            num_samples (int): Number of samples to display
            save_path (str): Path to save the plot
        """
        true_labels, predicted_labels, sample_images = self.predict_samples(num_samples)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(sample_images[i].squeeze(), cmap='gray')
            axes[i].set_title(f'True: {true_labels[i]}\nPred: {predicted_labels[i]}')
            axes[i].axis('off')
            
            # Color code based on correctness
            color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
            axes[i].title.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix for test set predictions.
        
        Args:
            save_path (str): Path to save the plot
        """
        # Get predictions for entire test set
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes))
    
    def save_model(self, model_path='models/mnist_cnn_model.h5'):
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/mnist_cnn_model.h5'):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def print_model_summary(self):
        """Print the model architecture summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("\nModel Architecture:")
        self.model.summary()


def main():
    """
    Main function to run the complete MNIST digit recognition pipeline.
    """
    print("=" * 60)
    print("MNIST Digit Recognition using Convolutional Neural Network")
    print("=" * 60)
    
    # Initialize the recognizer
    recognizer = MNISTDigitRecognizer()
    
    # Load and preprocess data
    recognizer.load_and_preprocess_data(validation_split=0.1)
    
    # Build the model
    recognizer.build_model()
    recognizer.print_model_summary()
    
    # Train the model
    recognizer.train_model(epochs=15, batch_size=128)
    
    # Evaluate the model
    test_loss, test_accuracy = recognizer.evaluate_model()
    
    # Save the model
    recognizer.save_model()
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Plot training history
    recognizer.plot_training_history('images/training_history.png')
    
    # Plot sample predictions
    recognizer.plot_sample_predictions(num_samples=10, 
                                     save_path='images/sample_predictions.png')
    
    # Plot confusion matrix
    recognizer.plot_confusion_matrix('images/confusion_matrix.png')
    
    print("\n" + "=" * 60)
    print("MNIST Digit Recognition Pipeline Completed Successfully!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
