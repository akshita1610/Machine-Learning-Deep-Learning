from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Interactive Image Classification Tool"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "A production-ready image classification tool with ML/DL capabilities"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # File Upload Configuration
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    
    # Model Configuration
    MODEL_NAME: str = "resnet50"
    MODEL_CACHE_DIR: str = "models"
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Image Processing Configuration
    MAX_IMAGE_SIZE: tuple = (1024, 1024)
    PREPROCESSING_SIZE: tuple = (224, 224)
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
