from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Interactive Image Classification Tool",
    description="A production-ready image classification tool with ML/DL capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Interactive Image Classification Tool API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "api_v1": "/api/v1"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": False,
        "current_model": "resnet50",
        "device": "cpu"
    }

@app.get("/api/v1/model/info")
async def get_model_info():
    """Get model information."""
    return {
        "name": "resnet50",
        "version": "1.0.0",
        "input_size": (224, 224),
        "num_classes": 1000,
        "device": "cpu",
        "description": "Pre-trained ResNet-50 model for image classification"
    }

@app.get("/api/v1/models")
async def get_available_models():
    """Get available models."""
    return {
        "models": [
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
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {
        "error": "HTTP Exception",
        "status_code": exc.status_code,
        "detail": exc.detail
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled Exception: {str(exc)}")
    return {
        "error": "Internal Server Error",
        "status_code": 500,
        "detail": "An unexpected error occurred"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
