from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import time
import logging

from backend.models.schemas import (
    ImageClassificationRequest, 
    ImageClassificationResponse,
    ModelInfo,
    ErrorResponse
)
from backend.services.classifier_service import ImageClassifier
from backend.services.file_service import file_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize classifier
classifier = ImageClassifier()

@router.post("/classify", response_model=ImageClassificationResponse)
async def classify_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = None
):
    """
    Classify uploaded images using the specified model.
    
    Args:
        files: List of image files to classify
        model_name: Optional model name override
    
    Returns:
        Classification results for all images
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Check file count limit
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files allowed.")
        
        # Validate each file
        for file in files:
            if not file_service.validate_file(file):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file: {file.filename}. Only image files are allowed."
                )
        
        # Save uploaded files
        file_paths = await file_service.save_upload_files(files)
        
        # Schedule cleanup of uploaded files
        background_tasks.add_task(
            lambda: [file_service.delete_file(path) for path in file_paths]
        )
        
        # Initialize classifier with specified model if provided
        if model_name and model_name != classifier.model_name:
            global classifier
            classifier = ImageClassifier(model_name)
        
        # Classify images
        start_time = time.time()
        results = classifier.classify_batch(file_paths)
        processing_time = time.time() - start_time
        
        # Format response
        response_data = []
        for result in results:
            if "error" in result:
                response_data.append({
                    "filename": file.name,
                    "success": False,
                    "error": result["error"]
                })
            else:
                response_data.append({
                    "filename": file.name,
                    "success": True,
                    "class_name": result["class_name"],
                    "confidence": result["confidence"],
                    "class_index": result["class_index"],
                    "processing_time": result["processing_time"]
                })
        
        return ImageClassificationResponse(
            success=True,
            results=response_data,
            model_used=classifier.model_name,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during classification")

@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the currently loaded model.
    
    Returns:
        Model information including name, version, and capabilities
    """
    try:
        model_info = classifier.get_model_info()
        return ModelInfo(**model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.get("/models")
async def get_available_models():
    """
    Get list of available models.
    
    Returns:
        List of available model names and descriptions
    """
    try:
        available_models = [
            {
                "name": "resnet50",
                "description": "ResNet-50: Deep residual network for image classification",
                "input_size": (224, 224),
                "num_classes": 1000
            },
            {
                "name": "resnet18",
                "description": "ResNet-18: Lightweight residual network",
                "input_size": (224, 224),
                "num_classes": 1000
            },
            {
                "name": "mobilenet_v2",
                "description": "MobileNetV2: Efficient mobile-friendly network",
                "input_size": (224, 224),
                "num_classes": 1000
            },
            {
                "name": "efficientnet_b0",
                "description": "EfficientNet-B0: Efficient and accurate network",
                "input_size": (224, 224),
                "num_classes": 1000
            }
        ]
        
        return {"models": available_models}
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get available models")

@router.post("/model/switch")
async def switch_model(model_name: str):
    """
    Switch to a different model.
    
    Args:
        model_name: Name of the model to switch to
    
    Returns:
        Success message with new model info
    """
    try:
        # Validate model name
        valid_models = ["resnet50", "resnet18", "mobilenet_v2", "efficientnet_b0"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model name. Available models: {', '.join(valid_models)}"
            )
        
        # Initialize new classifier
        global classifier
        classifier = ImageClassifier(model_name)
        
        model_info = classifier.get_model_info()
        
        return {
            "message": f"Successfully switched to {model_name}",
            "model_info": model_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to switch model")

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the service
    """
    return {
        "status": "healthy",
        "model_loaded": classifier.model is not None,
        "current_model": classifier.model_name,
        "device": str(classifier.device)
    }
