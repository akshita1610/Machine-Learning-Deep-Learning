"""
Our model definitions - both a custom CNN and a ResNet with transfer learning.
Pick whichever works best for your needs!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CustomCNN(nn.Module):
    """
    A simple CNN built from scratch for CIFAR-10.
    Good baseline that works reasonably well on 32x32 images.
    """
    
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # Three convolutional blocks with batch norm and pooling
        # Each block doubles the channels and halves the spatial size
        
        # Block 1: 3 -> 32 channels, 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Helps training stability
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 32 -> 64 channels, 16x16 -> 8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 64 -> 128 channels, 8x8 -> 4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers - 128 channels * 4x4 = 2048 features
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """Run the input through the network."""
        # Three conv blocks with ReLU and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and run through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # No softmax - CrossEntropyLoss handles that
        
        return x


class ResNetTransfer(nn.Module):
    """
    ResNet18 with transfer learning - great when you want better accuracy
    without training from scratch. We adapt it for 32x32 CIFAR images.
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNetTransfer, self).__init__()
        
        # Load ResNet18 (pretrained on ImageNet if specified)
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Adapt for 32x32 images - smaller conv kernel and remove maxpool
        # Original ResNet uses 7x7 conv for 224x224, we use 3x3 for 32x32
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Skip the maxpool for small images
        
        # Replace the final layer for CIFAR-10 (10 classes instead of 1000)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        """Just pass through ResNet."""
        return self.resnet(x)
    
    def freeze_base(self):
        """
        Freeze all layers except the final one.
        Useful if you only want to train the classifier head.
        """
        # Freeze everything except the final layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        for param in self.resnet.fc.parameters():
            param.requires_grad = True


def get_model(model_name='custom', num_classes=10, pretrained=True):
    """
    Create the model you want to use.
    
    Options:
        - 'custom': Our simple CNN (fast, fewer params)
        - 'resnet': ResNet18 with transfer learning (more accurate)
    """
    if model_name == 'custom':
        model = CustomCNN(num_classes=num_classes)
        print(f"Using CustomCNN model")
    elif model_name == 'resnet':
        model = ResNetTransfer(num_classes=num_classes, pretrained=pretrained)
        if pretrained:
            print(f"Using ResNet18 with pretrained weights (fine-tuning)")
        else:
            print(f"Using ResNet18 from scratch")
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'custom' or 'resnet'")
    
    # Show how many parameters we're working with
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Quick sanity check - make sure both models work
    print("=" * 50)
    print("Testing CustomCNN:")
    custom_model = get_model('custom', num_classes=10)
    
    test_input = torch.randn(4, 3, 32, 32)  # Batch of 4 images
    output = custom_model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output[0]}")
    
    print("\n" + "=" * 50)
    print("Testing ResNetTransfer (without pretrained):")
    resnet_model = get_model('resnet', num_classes=10, pretrained=False)
    
    output = resnet_model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output[0]}")
