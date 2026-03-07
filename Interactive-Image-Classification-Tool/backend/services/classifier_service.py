import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
import time
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class ImageClassifier:
    """Main image classification service using pre-trained models."""
    
    def __init__(self, model_name: str = None):
        """Initialize the classifier with specified model."""
        self.model_name = model_name or settings.MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.class_names = self._load_imagenet_classes()
        
        # Initialize model
        self._load_model()
        
    def _load_imagenet_classes(self) -> List[str]:
        """Load ImageNet class names."""
        # Simplified ImageNet classes for demonstration
        # In production, you'd load the full 1000 classes from a file
        return [f"class_{i}" for i in range(1000)]
    
    def _load_model(self):
        """Load the pre-trained model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            if self.model_name == "resnet50":
                self.model = models.resnet50(pretrained=True)
            elif self.model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
            elif self.model_name == "mobilenet_v2":
                self.model = models.mobilenet_v2(pretrained=True)
            elif self.model_name == "efficientnet_b0":
                self.model = models.efficientnet_b0(pretrained=True)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            self.model.eval()
            self.model.to(self.device)
            
            # Setup preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(settings.PREPROCESSING_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
            ])
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image for model inference."""
        try:
            # Load and validate image
            image = Image.open(image_path).convert('RGB')
            
            # Apply preprocessing transforms
            image_tensor = self.preprocess(image).unsqueeze(0)
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def classify_single_image(self, image_path: str) -> Dict[str, Any]:
        """Classify a single image and return results."""
        try:
            start_time = time.time()
            
            # Preprocess image
            input_tensor = self.preprocess_image(image_path)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            # Get top prediction
            top_prob, top_catid = torch.topk(probabilities, 1)
            
            processing_time = time.time() - start_time
            
            result = {
                "image_path": image_path,
                "class_name": self.class_names[top_catid[0].item()],
                "confidence": top_prob[0].item(),
                "class_index": top_catid[0].item(),
                "processing_time": processing_time
            }
            
            logger.info(f"Classified {image_path}: {result['class_name']} ({result['confidence']:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {str(e)}")
            return {
                "image_path": image_path,
                "error": str(e),
                "class_name": "error",
                "confidence": 0.0,
                "class_index": -1,
                "processing_time": 0.0
            }
    
    def classify_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple images in batch."""
        results = []
        
        for image_path in image_paths:
            result = self.classify_single_image(image_path)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "name": self.model_name,
            "version": "1.0.0",
            "input_size": settings.PREPROCESSING_SIZE,
            "num_classes": len(self.class_names),
            "device": str(self.device),
            "description": f"Pre-trained {self.model_name} model for image classification"
        }
    
    def validate_image(self, image_path: str) -> bool:
        """Validate if the image file is supported."""
        try:
            # Check file extension
            valid_extensions = settings.ALLOWED_EXTENSIONS
            if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
                return False
            
            # Try to open the image
            with Image.open(image_path) as img:
                img.verify()
            
            return True
            
        except Exception:
            return False
