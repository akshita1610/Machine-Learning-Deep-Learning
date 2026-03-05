from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import uvicorn

from config.settings import settings
from backend.controllers.classification_controller import router as classification_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(classification_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Interactive Image Classification Tool API",
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "api_v1": settings.API_V1_STR
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon."""
    return FileResponse("frontend/public/favicon.ico")

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
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
