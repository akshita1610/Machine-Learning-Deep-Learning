import os
import shutil
import aiofiles
from fastapi import UploadFile, HTTPException
from typing import List
import uuid
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FileService:
    """Service for handling file operations."""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
    
    async def save_upload_file(self, file: UploadFile) -> str:
        """Save an uploaded file and return the file path."""
        try:
            # Generate unique filename
            file_extension = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = self.upload_dir / unique_filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    async def save_upload_files(self, files: List[UploadFile]) -> List[str]:
        """Save multiple uploaded files and return file paths."""
        file_paths = []
        
        for file in files:
            file_path = await self.save_upload_file(file)
            file_paths.append(file_path)
        
        return file_paths
    
    def validate_file(self, file: UploadFile) -> bool:
        """Validate uploaded file."""
        # Check file extension
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return False
        
        # Check file size (10MB limit)
        if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
            return False
        
        return True
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file."""
        try:
            file_path = Path(file_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old files in upload directory."""
        try:
            import time
            current_time = time.time()
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_hours * 3600:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Global file service instance
file_service = FileService()
