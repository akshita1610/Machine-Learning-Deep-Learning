"""
Data loading and preprocessing for our image classifier.
Handles the CIFAR-10 dataset with some helpful augmentation.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


def get_data_loaders(data_dir='./data', batch_size=64, num_workers=2):
    """
    Set up the data loaders for training, validation, and testing.
    
    We use augmentation for training to help the model generalize better,
    but keep it simple for validation and testing.
    """
    
    # Training transformations - we augment the data to prevent overfitting
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # Random crops help with position invariance
        transforms.RandomHorizontalFlip(p=0.5),    # Flip images horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Vary colors a bit
        transforms.ToTensor(),                     # Convert PIL images to tensors
        transforms.Normalize(                      # Normalize using CIFAR-10 statistics
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    # Validation/Test transformations - just normalize, no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    # Download CIFAR-10 if it doesn't exist
    print("Loading CIFAR-10 dataset...")
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Split training data - 90% for training, 10% for validation
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Set seed for reproducibility
    )
    
    # Validation shouldn't use augmentation
    val_dataset.dataset.transform = test_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True         # Speed up data transfer if using GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # CIFAR-10 has 10 classes
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, class_names


def get_single_image(data_dir='./data', index=0):
    """
    Grab a single image from the test set - useful for debugging.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    image, label = dataset[index]
    return image, label, class_names[label]


if __name__ == "__main__":
    # Quick test to make sure everything loads correctly
    train_loader, val_loader, test_loader, class_names = get_data_loaders()
    
    # Get one batch and check the shapes
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")  # [batch_size, 3, 32, 32]
    print(f"Labels shape: {labels.shape}")   # [batch_size]
    print(f"Sample labels: {[class_names[label] for label in labels[:5]]}")
