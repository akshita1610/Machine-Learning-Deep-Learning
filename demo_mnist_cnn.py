"""
Demo script that creates synthetic MNIST-like data for testing the CNN implementation.
This bypasses the OpenML download issues while demonstrating the complete pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class SyntheticMNISTGenerator:
    """Generate synthetic MNIST-like digit images for demonstration."""
    
    def __init__(self):
        self.digit_patterns = self._create_digit_patterns()
    
    def _create_digit_patterns(self):
        """Create simple patterns for digits 0-9."""
        patterns = {}
        
        # Simple 7x7 patterns for each digit
        patterns[0] = np.array([
            [0,1,1,1,1,1,0],
            [1,1,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [0,1,1,1,1,1,0]
        ])
        
        patterns[1] = np.array([
            [0,0,0,1,0,0,0],
            [0,0,1,1,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,1,0,0,0],
            [0,1,1,1,1,1,0]
        ])
        
        patterns[2] = np.array([
            [0,1,1,1,1,1,0],
            [1,1,0,0,0,1,1],
            [0,0,0,0,0,1,1],
            [0,0,0,0,1,1,0],
            [0,0,0,1,1,0,0],
            [0,0,1,1,0,0,0],
            [1,1,1,1,1,1,1]
        ])
        
        patterns[3] = np.array([
            [0,1,1,1,1,1,0],
            [1,1,0,0,0,1,1],
            [0,0,0,0,0,1,1],
            [0,0,1,1,1,1,0],
            [0,0,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [0,1,1,1,1,1,0]
        ])
        
        patterns[4] = np.array([
            [0,0,0,0,1,1,0],
            [0,0,0,1,1,1,0],
            [0,0,1,1,0,1,0],
            [0,1,1,0,0,1,0],
            [1,1,1,1,1,1,1],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,1,0]
        ])
        
        patterns[5] = np.array([
            [1,1,1,1,1,1,1],
            [1,1,0,0,0,0,0],
            [1,1,1,1,1,1,0],
            [0,0,0,0,0,1,1],
            [0,0,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [0,1,1,1,1,1,0]
        ])
        
        patterns[6] = np.array([
            [0,1,1,1,1,1,0],
            [1,1,0,0,0,0,0],
            [1,1,1,1,1,1,0],
            [1,1,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [0,1,1,1,1,1,0]
        ])
        
        patterns[7] = np.array([
            [1,1,1,1,1,1,1],
            [0,0,0,0,0,1,1],
            [0,0,0,0,1,1,0],
            [0,0,0,1,1,0,0],
            [0,0,1,1,0,0,0],
            [0,0,1,0,0,0,0],
            [0,1,0,0,0,0,0]
        ])
        
        patterns[8] = np.array([
            [0,1,1,1,1,1,0],
            [1,1,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [0,1,1,1,1,1,0],
            [1,1,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [0,1,1,1,1,1,0]
        ])
        
        patterns[9] = np.array([
            [0,1,1,1,1,1,0],
            [1,1,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [0,1,1,1,1,1,1],
            [0,0,0,0,0,1,1],
            [1,1,0,0,0,1,1],
            [0,1,1,1,1,1,0]
        ])
        
        return patterns
    
    def generate_digit(self, digit, size=28, noise_level=0.1):
        """
        Generate a synthetic digit image.
        
        Args:
            digit (int): Digit to generate (0-9)
            size (int): Image size (size x size)
            noise_level (float): Amount of random noise to add
            
        Returns:
            numpy.ndarray: Generated digit image
        """
        if digit not in self.digit_patterns:
            raise ValueError(f"Digit {digit} not supported. Use 0-9.")
        
        # Get the base pattern
        pattern = self.digit_patterns[digit]
        
        # Scale up to desired size
        from scipy.ndimage import zoom
        scale_factor = size / 7
        scaled_pattern = zoom(pattern, scale_factor, order=1)
        
        # Add noise
        noise = np.random.normal(0, noise_level, scaled_pattern.shape)
        noisy_image = scaled_pattern + noise
        
        # Clip to [0, 1] range
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image
    
    def generate_dataset(self, samples_per_digit=1000, size=28, noise_level=0.1):
        """
        Generate a complete synthetic MNIST dataset.
        
        Args:
            samples_per_digit (int): Number of samples per digit
            size (int): Image size
            noise_level (float): Noise level
            
        Returns:
            tuple: (X, y) where X is images and y is labels
        """
        X = []
        y = []
        
        for digit in range(10):
            for _ in range(samples_per_digit):
                image = self.generate_digit(digit, size, noise_level)
                X.append(image)
                y.append(digit)
        
        return np.array(X), np.array(y)


class SimpleCNN:
    """
    A simplified CNN implementation for MNIST digit recognition.
    This is a basic implementation for educational purposes.
    """
    
    def __init__(self):
        """Initialize the Simple CNN."""
        self.filters = None
        self.weights = None
        self.biases = None
        self.training_history = {'loss': [], 'accuracy': []}
        
    def load_synthetic_data(self, samples_per_digit=1000, test_size=0.2, random_state=42):
        """
        Load synthetic MNIST dataset.
        
        Args:
            samples_per_digit (int): Number of samples per digit
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        print("Loading synthetic MNIST dataset...")
        
        # Generate synthetic data
        generator = SyntheticMNISTGenerator()
        X, y = generator.generate_dataset(samples_per_digit, size=28, noise_level=0.1)
        
        # Reshape to 28x28x1
        X = X.reshape(-1, 28, 28, 1)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # One-hot encode labels
        encoder = OneHotEncoder(sparse_output=False)
        y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train_encoded, y_test_encoded, y_train, y_test
    
    def initialize_parameters(self, input_shape=(28, 28, 1), num_classes=10):
        """
        Initialize network parameters.
        
        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
        """
        # Simple convolution filter (3x3)
        self.num_filters = 8
        self.filter_size = 3
        self.filters = np.random.randn(self.num_filters, self.filter_size, self.filter_size) * 0.1
        
        # Fully connected layer parameters
        # After convolution and pooling: (28-2)//2 = 13, so 13*13*8 = 1352
        self.fc_input_size = 13 * 13 * self.num_filters
        self.weights = np.random.randn(self.fc_input_size, num_classes) * 0.1
        self.biases = np.zeros((1, num_classes))
        
        print(f"Initialized {self.num_filters} filters of size {self.filter_size}x{self.filter_size}")
        print(f"Fully connected layer: {self.fc_input_size} -> {num_classes}")
    
    def convolution(self, image):
        """
        Perform convolution operation.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Convolved feature maps
        """
        height, width = image.shape[:2]
        convolved = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        
        for i in range(self.num_filters):
            for h in range(height - self.filter_size + 1):
                for w in range(width - self.filter_size + 1):
                    convolved[h, w, i] = np.sum(image[h:h+self.filter_size, w:w+self.filter_size, 0] * self.filters[i])
        
        return convolved
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def max_pooling(self, feature_map, pool_size=2):
        """
        Perform max pooling operation.
        
        Args:
            feature_map (numpy.ndarray): Input feature map
            pool_size (int): Size of pooling window
            
        Returns:
            numpy.ndarray: Pooled feature map
        """
        height, width, num_filters = feature_map.shape
        pooled_height = height // pool_size
        pooled_width = width // pool_size
        pooled = np.zeros((pooled_height, pooled_width, num_filters))
        
        for i in range(num_filters):
            for h in range(pooled_height):
                for w in range(pooled_width):
                    pooled[h, w, i] = np.max(feature_map[h*pool_size:(h+1)*pool_size, w*pool_size:(w+1)*pool_size, i])
        
        return pooled
    
    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, image):
        """
        Forward pass through the network.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Network output probabilities
        """
        # Convolution
        convolved = self.convolution(image)
        
        # ReLU activation
        activated = self.relu(convolved)
        
        # Max pooling
        pooled = self.max_pooling(activated)
        
        # Flatten for fully connected layer
        flattened = pooled.reshape(1, -1)
        
        # Fully connected layer
        logits = np.dot(flattened, self.weights) + self.biases
        
        # Softmax activation
        output = self.softmax(logits)
        
        return output, flattened
    
    def compute_loss(self, predictions, targets):
        """
        Compute cross-entropy loss.
        
        Args:
            predictions (numpy.ndarray): Model predictions
            targets (numpy.ndarray): True labels (one-hot encoded)
            
        Returns:
            float: Cross-entropy loss
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        loss = -np.mean(np.sum(targets * np.log(predictions), axis=1))
        return loss
    
    def compute_accuracy(self, predictions, targets):
        """
        Compute accuracy.
        
        Args:
            predictions (numpy.ndarray): Model predictions
            targets (numpy.ndarray): True labels (one-hot encoded)
            
        Returns:
            float: Accuracy
        """
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
        return np.mean(pred_classes == true_classes)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, learning_rate=0.01):
        """
        Train the CNN using simple gradient descent.
        
        Args:
            X_train (numpy.ndarray): Training images
            y_train (numpy.ndarray): Training labels (one-hot encoded)
            X_val (numpy.ndarray): Validation images
            y_val (numpy.ndarray): Validation labels (one-hot encoded)
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
        """
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Simple training: process a subset of training data
            batch_size = min(100, len(X_train))
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            for i, idx in enumerate(indices):
                image = X_train[idx]
                target = y_train[idx:idx+1]
                
                # Forward pass
                output, flattened = self.forward_pass(image)
                
                # Compute loss and accuracy
                loss = self.compute_loss(output, target)
                accuracy = self.compute_accuracy(output, target)
                
                epoch_loss += loss
                epoch_accuracy += accuracy
                
                # Simple gradient update (simplified for educational purposes)
                error = output - target
                grad_weights = flattened.T @ error
                grad_biases = np.sum(error, axis=0, keepdims=True)
                
                # Update parameters
                self.weights -= learning_rate * grad_weights
                self.biases -= learning_rate * grad_biases
            
            # Validation
            val_loss = 0
            val_accuracy = 0
            val_batch_size = min(50, len(X_val))
            val_indices = np.random.choice(len(X_val), val_batch_size, replace=False)
            
            for idx in val_indices:
                image = X_val[idx]
                target = y_val[idx:idx+1]
                
                output, _ = self.forward_pass(image)
                val_loss += self.compute_loss(output, target)
                val_accuracy += self.compute_accuracy(output, target)
            
            avg_train_loss = epoch_loss / batch_size
            avg_train_acc = epoch_accuracy / batch_size
            avg_val_loss = val_loss / val_batch_size
            avg_val_acc = val_accuracy / val_batch_size
            
            self.training_history['loss'].append(avg_train_loss)
            self.training_history['accuracy'].append(avg_train_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X (numpy.ndarray): Input images
            
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        predictions = []
        for image in X:
            output, _ = self.forward_pass(image)
            predictions.append(output[0])
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (numpy.ndarray): Test images
            y_test (numpy.ndarray): Test labels (one-hot encoded)
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        predictions = self.predict(X_test)
        loss = self.compute_loss(predictions, y_test)
        accuracy = self.compute_accuracy(predictions, y_test)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.training_history['loss'], label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.training_history['accuracy'], label='Training Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_sample_predictions(self, X_test, y_test, y_test_original, num_samples=10, save_path=None):
        """Plot sample predictions with true labels."""
        predictions = self.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Select random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(X_test[idx].squeeze(), cmap='gray')
            true_label = y_test_original[idx]
            pred_label = pred_classes[idx]
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}')
            axes[i].axis('off')
            
            # Color code based on correctness
            color = 'green' if true_label == pred_label else 'red'
            axes[i].title.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, X_test, y_test, y_test_original, save_path=None):
        """Plot confusion matrix."""
        predictions = self.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        cm = confusion_matrix(y_test_original, pred_classes)
        
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
        print(classification_report(y_test_original, pred_classes))


def main():
    """Main function to run the simplified MNIST CNN pipeline."""
    print("=" * 60)
    print("Synthetic MNIST Digit Recognition using CNN")
    print("=" * 60)
    
    # Initialize the CNN
    cnn = SimpleCNN()
    
    # Load synthetic data
    X_train, X_test, y_train, y_test, y_train_original, y_test_original = cnn.load_synthetic_data(samples_per_digit=500)
    
    # Split training data into train and validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize parameters
    cnn.initialize_parameters()
    
    # Train the model
    cnn.train(X_train_split, y_train_split, X_val_split, y_val_split, epochs=15, learning_rate=0.01)
    
    # Evaluate on test set
    test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot training history
    cnn.plot_training_history('images/training_history.png')
    
    # Plot sample predictions
    cnn.plot_sample_predictions(X_test, y_test, y_test_original, num_samples=10, 
                               save_path='images/sample_predictions.png')
    
    # Plot confusion matrix
    cnn.plot_confusion_matrix(X_test, y_test, y_test_original, 'images/confusion_matrix.png')
    
    print("\n" + "=" * 60)
    print("Synthetic MNIST CNN Pipeline Completed!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
