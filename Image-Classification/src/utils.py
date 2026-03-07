"""
Helper functions for saving checkpoints, creating visualizations, and computing metrics.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


def save_checkpoint(epoch, model, optimizer, best_acc, filepath):
    """
    Save a checkpoint so we can resume training later or use the best model.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load a checkpoint back into the model (and optionally optimizer).
    
    Returns the epoch we left off at and the best accuracy so far.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_acc']


def plot_training_curves(history, save_path):
    """
    Create a nice plot showing how loss and accuracy changed during training.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue', marker='o')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Create a heatmap showing where the model gets confused between classes.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Plot normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Percentages)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sample_predictions(images, labels, predictions, class_names, save_path, num_samples=16):
    """
    Show a grid of images with their true labels and what the model predicted.
    """
    # Denormalize images for display
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    
    # Convert to numpy and denormalize
    images = images.cpu().numpy()
    images = images * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
    images = np.clip(images, 0, 1)
    
    # Create subplot grid
    num_samples = min(num_samples, len(images))
    num_rows = int(np.ceil(np.sqrt(num_samples)))
    num_cols = int(np.ceil(num_samples / num_rows))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Plot image
        img = np.transpose(images[i], (1, 2, 0))  # CHW -> HWC
        ax.imshow(img)
        
        # Determine color based on correct/incorrect
        is_correct = labels[i] == predictions[i]
        color = 'green' if is_correct else 'red'
        
        # Set title
        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_classification_report(y_true, y_pred, class_names):
    """
    Print a detailed report with precision, recall, and F1-score for each class.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print("=" * 60)
    print(report)
    print("=" * 60)


def set_seed(seed=42):
    """
    Set random seeds for reproducibility. Use the same seed to get the same results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count how many parameters are in the model (total and trainable).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
