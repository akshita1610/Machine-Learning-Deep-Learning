# Development Guide

This guide provides detailed information for developers working on the Interactive Image Classification Tool.

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- Git
- VS Code (recommended)

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd Interactive-Image-Classification-Tool

# Backend setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
cd ..

# Development servers
# Terminal 1: Backend
python main.py

# Terminal 2: Frontend
cd frontend && npm start
```

## 🏗️ Project Structure

```
Interactive-Image-Classification-Tool/
├── backend/                    # FastAPI backend
│   ├── controllers/           # API endpoints
│   │   └── classification_controller.py
│   ├── services/              # Business logic
│   │   ├── classifier_service.py
│   │   └── file_service.py
│   ├── models/                # Data models
│   │   └── schemas.py
│   └── api/                   # API routes (if needed)
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── ImageUpload.jsx
│   │   │   ├── ClassificationResults.jsx
│   │   │   └── ModelSelector.jsx
│   │   ├── services/           # API integration
│   │   │   └── api.js
│   │   ├── utils/              # Helper functions
│   │   │   └── helpers.js
│   │   ├── App.jsx            # Main application
│   │   └── index.js           # Entry point
│   ├── public/                # Static assets
│   └── package.json
├── config/                     # Configuration
│   └── settings.py
├── utils/                      # Shared utilities
├── docs/                       # Documentation
├── uploads/                    # Temporary file storage
├── requirements.txt            # Python dependencies
├── main.py                     # FastAPI application entry
└── README.md                   # Project documentation
```

## 🔧 Backend Development

### Adding New Models
1. Update `classifier_service.py`:
```python
def _load_model(self):
    if self.model_name == "your_new_model":
        self.model = models.your_new_model(pretrained=True)
        # Add preprocessing transforms
```

2. Update available models list in `classification_controller.py`:
```python
available_models = [
    # ... existing models
    {
        "name": "your_new_model",
        "description": "Your model description",
        "input_size": (224, 224),
        "num_classes": 1000
    }
]
```

### Adding New API Endpoints
1. Create new controller in `backend/controllers/`
2. Define schemas in `backend/models/schemas.py`
3. Register router in `main.py`

### Error Handling
```python
try:
    # Your code
except SpecificException as e:
    logger.error(f"Specific error: {str(e)}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

## 🎨 Frontend Development

### Component Structure
```jsx
// Component template
import React, { useState, useEffect } from 'react';
import { iconLibrary } from 'lucide-react';

const ComponentName = ({ prop1, prop2 }) => {
  const [state, setState] = useState(initialValue);
  
  useEffect(() => {
    // Side effects
  }, [dependencies]);

  const handleAction = () => {
    // Event handlers
  };

  return (
    <div className="component-styles">
      {/* JSX content */}
    </div>
  );
};

export default ComponentName;
```

### Adding New Components
1. Create component file in `frontend/src/components/`
2. Follow naming convention: `PascalCase.jsx`
3. Use TailwindCSS for styling
4. Export as default

### API Integration
```javascript
import { imageClassificationAPI } from '../services/api';

const YourComponent = () => {
  const handleApiCall = async () => {
    try {
      const response = await imageClassificationAPI.yourMethod();
      // Handle response
    } catch (error) {
      // Handle error
    }
  };
};
```

## 🧪 Testing

### Backend Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html
```

### Frontend Tests
```bash
# Run unit tests
npm test

# Run coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

### Test Structure
```
tests/
├── backend/
│   ├── test_classifier_service.py
│   ├── test_file_service.py
│   └── test_api_endpoints.py
└── frontend/
    ├── components/
    └── services/
```

## 📝 Code Style

### Python (Backend)
- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Use f-strings for string formatting

```python
def classify_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Classify multiple images and return results.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of classification results
    """
    # Implementation
```

### JavaScript (Frontend)
- Use ESLint and Prettier
- Prefer functional components with hooks
- Use descriptive variable names
- Comment complex logic

```javascript
// Good
const [isLoading, setIsLoading] = useState(false);

// Bad
const [l, setL] = useState(false);
```

## 🚀 Deployment

### Docker Development
```bash
# Build for development
docker build -f Dockerfile.dev -t image-classifier-dev .

# Run development container
docker run -p 8000:8000 -v $(pwd):/app image-classifier-dev
```

### Production Build
```bash
# Backend
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# Frontend
cd frontend
npm run build
```

## 🔍 Debugging

### Backend Debugging
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints
logger.debug(f"Debug info: {variable}")
```

### Frontend Debugging
```javascript
// Use React DevTools
// Add console.log for debugging
console.log('Debug info:', data);

// Use debugger statement
debugger;
```

## 📊 Performance Optimization

### Backend
- Use async/await for I/O operations
- Implement caching for model loading
- Use connection pooling for database operations
- Monitor memory usage

### Frontend
- Use React.memo for expensive components
- Implement lazy loading for large components
- Optimize images and assets
- Use code splitting

## 🔐 Security Considerations

### File Upload Security
- Validate file types and sizes
- Scan uploaded files for malware
- Use secure file storage
- Implement rate limiting

### API Security
- Validate all input parameters
- Use HTTPS in production
- Implement authentication if needed
- Sanitize error messages

## 📈 Monitoring

### Backend Metrics
- Request response times
- Error rates
- Memory usage
- Model inference times

### Frontend Metrics
- Page load times
- User interaction events
- Error tracking
- Performance metrics

## 🤝 Contributing Guidelines

1. **Branch Naming**
   - `feature/feature-name`
   - `bugfix/bug-description`
   - `hotfix/critical-fix`

2. **Commit Messages**
   - Use conventional commits
   - `feat: add new feature`
   - `fix: resolve bug`
   - `docs: update documentation`

3. **Pull Requests**
   - Include tests for new features
   - Update documentation
   - Ensure CI passes
   - Request code review

## 📚 Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [TailwindCSS Documentation](https://tailwindcss.com/docs)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Tools
- [Postman](https://www.postman.com/) for API testing
- [React DevTools](https://react.dev/learn/react-developer-tools)
- [Python Debugging](https://docs.python.org/3/library/pdb.html)

---

Happy coding! 🚀
