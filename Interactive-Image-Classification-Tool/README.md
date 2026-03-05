# Interactive Image Classification Tool

A production-ready, web-based interactive image classification tool built with modern machine learning and web technologies. This project demonstrates clean software engineering practices inspired by industry standards.

## 🚀 Features

### Core Functionality
- **Multi-image Upload**: Drag-and-drop interface supporting up to 10 images simultaneously
- **Real-time Classification**: Fast inference using pre-trained deep learning models
- **Multiple Models**: Choose from ResNet, MobileNet, and EfficientNet architectures
- **Confidence Scoring**: Detailed confidence scores and class predictions
- **Batch Processing**: Classify multiple images in a single request
- **Results Export**: Download classification results in JSON or CSV format

### Advanced Features
- **Model Switching**: Dynamically switch between different ML models
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Responsive Design**: Mobile-friendly interface that works on all devices
- **Real-time Feedback**: Loading states, progress indicators, and toast notifications
- **File Validation**: Automatic file type and size validation
- **Class Distribution**: Visual analytics of classification results

## 🏗️ Architecture

### Backend (FastAPI + PyTorch)
```
backend/
├── controllers/          # API endpoints and request handling
├── services/           # Business logic and ML pipeline
├── models/             # Data models and schemas
└── config/             # Configuration management
```

### Frontend (React + TailwindCSS)
```
frontend/
├── src/
│   ├── components/     # Reusable React components
│   ├── services/       # API integration
│   └── utils/          # Helper functions
└── public/             # Static assets
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework for model inference
- **Torchvision**: Pre-trained models and image processing
- **Pillow**: Image processing and manipulation
- **Uvicorn**: ASGI server for production deployment

### Frontend
- **React 18**: Modern UI library with hooks
- **TailwindCSS**: Utility-first CSS framework
- **Axios**: HTTP client for API communication
- **React Dropzone**: File upload component
- **React Toastify**: Notification system
- **Lucide React**: Modern icon library

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- Git

### Backend Setup
```bash
# Navigate to project directory
cd Interactive-Image-Classification-Tool

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at `http://localhost:3000`

## 🎯 Usage

### Basic Usage
1. **Upload Images**: Drag and drop images or click to select files
2. **Select Model**: Choose from available ML models (ResNet50, MobileNetV2, etc.)
3. **Classify**: Click the "Classify Images" button to process
4. **View Results**: See predictions, confidence scores, and processing time
5. **Export**: Download results in JSON or CSV format

### Advanced Usage
- **Model Comparison**: Switch between models to compare results
- **Batch Processing**: Upload multiple images for efficient processing
- **Error Analysis**: Review failed classifications with detailed error messages

## 📊 Available Models

| Model | Description | Input Size | Classes | Speed |
|-------|-------------|------------|---------|-------|
| ResNet50 | Deep residual network | 224×224 | 1000 | Medium |
| ResNet18 | Lightweight residual network | 224×224 | 1000 | Fast |
| MobileNetV2 | Mobile-friendly architecture | 224×224 | 1000 | Very Fast |
| EfficientNet-B0 | Efficient and accurate | 224×224 | 1000 | Fast |

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
# API Configuration
API_V1_STR=/api/v1
PROJECT_NAME="Interactive Image Classification Tool"
DEBUG=false

# Server Configuration
HOST=0.0.0.0
PORT=8000

# File Upload Configuration
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=.jpg,.jpeg,.png,.bmp,.tiff,.webp

# Model Configuration
MODEL_NAME=resnet50
CONFIDENCE_THRESHOLD=0.5
```

## 🧪 Testing

### Backend Tests
```bash
# Run API tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=backend
```

### Frontend Tests
```bash
# Run unit tests
npm test

# Run end-to-end tests
npm run test:e2e
```

## 📈 Performance

### Benchmarks
- **Single Image Classification**: ~50ms (ResNet50 on CPU)
- **Batch Processing**: ~200ms for 10 images
- **Model Switching**: ~2s (first time only)
- **File Upload**: Supports up to 10MB per image

### Optimization Features
- **Model Caching**: Models are cached after first load
- **Batch Processing**: Efficient handling of multiple images
- **Memory Management**: Automatic cleanup of uploaded files
- **Async Processing**: Non-blocking file operations

## 🚀 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t image-classifier .

# Run container
docker run -p 8000:8000 image-classifier
```

### Production Deployment
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Build frontend for production
cd frontend
npm run build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint and Prettier for JavaScript/React code
- Write comprehensive tests for new features
- Update documentation for API changes

## 📝 API Documentation

### Endpoints

#### Classify Images
```
POST /api/v1/classify
Content-Type: multipart/form-data

Parameters:
- files: Image files (max 10)
- model_name: Optional model override

Response:
{
  "success": true,
  "results": [...],
  "model_used": "resnet50",
  "processing_time": 1.23
}
```

#### Get Model Info
```
GET /api/v1/model/info

Response:
{
  "name": "resnet50",
  "version": "1.0.0",
  "input_size": [224, 224],
  "num_classes": 1000,
  "description": "..."
}
```

#### Available Models
```
GET /api/v1/models

Response:
{
  "models": [
    {
      "name": "resnet50",
      "description": "...",
      "input_size": [224, 224],
      "num_classes": 1000
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
- **Issue**: Model fails to load
- **Solution**: Check internet connection for first-time download
- **Log**: Check backend logs for detailed error messages

#### File Upload Errors
- **Issue**: Files not uploading
- **Solution**: Verify file format and size limits
- **Check**: Ensure file extensions are supported

#### Frontend Connection Issues
- **Issue**: Frontend can't connect to backend
- **Solution**: Ensure backend is running on port 8000
- **Check**: CORS configuration in settings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) for the modern web framework
- [React](https://reactjs.org/) for the UI library
- [TailwindCSS](https://tailwindcss.com/) for the utility-first CSS framework

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the [documentation](docs/)
- Review the [FAQ](docs/FAQ.md)

---

**Built with ❤️ using modern ML and web technologies**
