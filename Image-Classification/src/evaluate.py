"""
Evaluation script - tests the trained model and creates visualizations.
"""

import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from tqdm import tqdm

from model import get_model
from data_loader import get_data_loaders
from utils import (
    load_checkpoint, 
    plot_confusion_matrix, 
    plot_sample_predictions,
    print_classification_report
)


def evaluate_model(model, test_loader, device):
    """
    Test the model on the test dataset.
    
    Returns loss, accuracy, and all predictions/labels for analysis.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_images = []
    
    print("Evaluating on test set...")
    
    # No gradients needed during evaluation
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Save predictions for later visualization
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.append(images.cpu())
            
            # Show progress
            acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    # Combine all images into one tensor
    all_images = torch.cat(all_images, dim=0)
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels), all_images


def evaluate(args):
    """
    Main evaluation function - loads model and runs full test.
    """
    print("=" * 60)
    print("Image Classification Evaluation")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading test dataset...")
    _, _, test_loader, class_names = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print(f"\nInitializing model: {args.model}")
    model = get_model(
        model_name=args.model,
        num_classes=10,
        pretrained=False  # Don't need pretrained for evaluation
    )
    model = model.to(device)
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    print(f"Loading model from: {args.model_path}")
    _, _ = load_checkpoint(args.model_path, model)
    
    # Create output directory for results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    test_loss, test_acc, predictions, labels, images = evaluate_model(
        model, test_loader, device
    )
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("=" * 60)
    
    # Classification report
    print_classification_report(labels, predictions, class_names)
    
    # Create confusion matrix visualization
    print("\nNow, let's take a look at how our model performed on each class. We'll create a confusion matrix to visualize this.")
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(labels, predictions, class_names, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Create sample predictions grid
    print("\nNext, let's see how our model performed on some sample images. We'll create a grid of predictions to visualize this.")
    sample_size = min(16, len(images))
    sample_images = images[:sample_size]
    sample_labels = labels[:sample_size]
    sample_predictions = predictions[:sample_size]
    
    pred_path = os.path.join(args.output_dir, 'sample_predictions.png')
    plot_sample_predictions(
        sample_images, sample_labels, sample_predictions, 
        class_names, pred_path, num_samples=sample_size
    )
    print(f"Sample predictions saved to: {pred_path}")
    
    # Save results to file
    results_path = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("Image Classification Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Model Path: {args.model_path}\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write("=" * 60 + "\n")
        
        from sklearn.metrics import classification_report
        report = classification_report(labels, predictions, target_names=class_names, digits=4)
        f.write(report)
    
    print(f"\nResults saved to: {results_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained image classification model')
    
    # Model options
    parser.add_argument('--model', type=str, default='custom',
                        choices=['custom', 'resnet'],
                        help='Which model architecture was used')
    parser.add_argument('--model_path', type=str, 
                        default='./outputs/checkpoints/best_model.pth',
                        help='Path to the trained model')
    
    # Data options
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Where the dataset is stored')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation',
                        help='Where to save evaluation results')
    
    args = parser.parse_args()
    
    # Show configuration
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # Run evaluation
    evaluate(args)


if __name__ == "__main__":
    main()
