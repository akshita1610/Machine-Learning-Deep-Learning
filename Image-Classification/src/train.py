"""
Training script - handles the main training loop with validation and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
import json
import time
from tqdm import tqdm

from model import get_model
from data_loader import get_data_loaders
from utils import save_checkpoint, load_checkpoint, plot_training_curves


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Run one training epoch.
    
    Returns average loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar makes it easier to see training progress
    pbar = tqdm(train_loader, desc='Training')
    
    for images, labels in pbar:
        # Move data to GPU/CPU
        images, labels = images.to(device), labels.to(device)
        
        # Clear gradients from previous step
        optimizer.zero_grad()
        
        # Forward pass - get predictions
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass - calculate gradients
        loss.backward()
        
        # Update model weights
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Show live stats in progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    """
    Run validation on the validation set.
    
    No gradient computation here since we're just evaluating.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Don't compute gradients during validation (saves memory and time)
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train(args):
    """
    Main training function - orchestrates the whole training process.
    """
    print("=" * 60)
    print("Image Classification Training")
    print("=" * 60)
    
    # Pick device - use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for saving results
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Load up the CIFAR-10 dataset
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create the model
    print("\nInitializing model...")
    model = get_model(
        model_name=args.model,
        num_classes=10,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Loss function and optimizer setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler - reduces LR when validation loss stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Keep track of training history for plotting later
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_epoch = 0
    
    # Load from checkpoint if resuming training
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, best_val_acc = load_checkpoint(args.resume, model, optimizer)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("-" * 60)
    
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate based on validation performance
        scheduler.step(val_loss)
        
        # Save this epoch's results
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
            save_checkpoint(epoch, model, optimizer, best_val_acc, checkpoint_path)
            print(f"  + Saved best model (Val Acc: {best_val_acc:.2f}%)")
        
        # Also save the latest model (in case we want to resume)
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'latest_model.pth')
        save_checkpoint(epoch, model, optimizer, best_val_acc, checkpoint_path)
    
    # Save training history as JSON for later analysis
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining history saved to {history_path}")
    
    # Create training curves plot
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(history, plot_path)
    print(f"Training curves saved to {plot_path}")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train an image classification model')
    
    # Model options
    parser.add_argument('--model', type=str, default='custom', 
                        choices=['custom', 'resnet'],
                        help='Which model to use (custom or resnet)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights (for ResNet)')
    
    # Data options
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Where to store the dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='How many images per batch')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=30,
                        help='How many epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    
    # Checkpoint options
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Where to save results')
    
    args = parser.parse_args()
    
    # Show what we're doing
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()
