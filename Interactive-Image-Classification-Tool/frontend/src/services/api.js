import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`Response received: ${response.status}`);
    return response;
  },
  (error) => {
    console.error('Response error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API service methods
export const imageClassificationAPI = {
  // Classify images
  classifyImages: async (files, modelName = null) => {
    const formData = new FormData();
    
    files.forEach((file) => {
      formData.append('files', file);
    });
    
    if (modelName) {
      formData.append('model_name', modelName);
    }
    
    const response = await api.post('/classify', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  // Get model information
  getModelInfo: async () => {
    const response = await api.get('/model/info');
    return response.data;
  },

  // Get available models
  getAvailableModels: async () => {
    const response = await api.get('/models');
    return response.data;
  },

  // Switch model
  switchModel: async (modelName) => {
    const response = await api.post('/model/switch', null, {
      params: { model_name: modelName },
    });
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;
