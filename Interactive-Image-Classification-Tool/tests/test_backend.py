import pytest
import tempfile
import os
from PIL import Image
import io
from backend.services.classifier_service import ImageClassifier
from backend.services.file_service import FileService

class TestImageClassifier:
    """Test suite for ImageClassifier service."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance for testing."""
        return ImageClassifier("resnet18")  # Use lighter model for testing
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple test image
        image = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.model_name == "resnet18"
        assert classifier.model is not None
        assert classifier.preprocess is not None
        assert len(classifier.class_names) == 1000
    
    def test_model_info(self, classifier):
        """Test model info retrieval."""
        info = classifier.get_model_info()
        assert "name" in info
        assert "version" in info
        assert "input_size" in info
        assert "num_classes" in info
        assert info["name"] == "resnet18"
    
    def test_image_validation(self, classifier):
        """Test image validation."""
        # Test valid image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image = Image.new('RGB', (224, 224), color='blue')
            image.save(tmp.name)
            assert classifier.validate_image(tmp.name) == True
            os.unlink(tmp.name)
        
        # Test invalid extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"not an image")
            assert classifier.validate_image(tmp.name) == False
            os.unlink(tmp.name)

class TestFileService:
    """Test suite for FileService."""
    
    @pytest.fixture
    def file_service(self):
        """Create FileService instance for testing."""
        return FileService("test_uploads")
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing."""
        image = Image.new('RGB', (100, 100), color='green')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        from fastapi import UploadFile
        return UploadFile(
            filename="test.png",
            file=img_byte_arr,
            content_type="image/png"
        )
    
    def test_file_validation(self, file_service, sample_image_file):
        """Test file validation."""
        assert file_service.validate_file(sample_image_file) == True
        
        # Test invalid file type
        invalid_file = UploadFile(
            filename="test.txt",
            file=io.BytesIO(b"not an image"),
            content_type="text/plain"
        )
        assert file_service.validate_file(invalid_file) == False

# Integration tests
class TestClassificationIntegration:
    """Integration tests for the complete classification pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_classification_pipeline(self):
        """Test the complete classification pipeline."""
        # This would require setting up the full FastAPI app
        # and testing the API endpoints
        pass

# Performance tests
class TestPerformance:
    """Performance tests for the classification service."""
    
    def test_classification_speed(self):
        """Test classification performance."""
        classifier = ImageClassifier("resnet18")
        
        # Create test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image = Image.new('RGB', (224, 224), color='red')
            image.save(tmp.name)
            
            import time
            start_time = time.time()
            result = classifier.classify_single_image(tmp.name)
            end_time = time.time()
            
            # Classification should complete within reasonable time
            assert end_time - start_time < 5.0  # 5 seconds max
            assert "class_name" in result
            assert "confidence" in result
            
            os.unlink(tmp.name)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
