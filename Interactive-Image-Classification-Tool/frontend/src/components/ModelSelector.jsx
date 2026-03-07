import React, { useState, useEffect } from 'react';
import { Settings, Brain, CheckCircle } from 'lucide-react';
import { imageClassificationAPI } from '../services/api';

const ModelSelector = ({ selectedModel, onModelChange, isLoading = false }) => {
  const [models, setModels] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);

  useEffect(() => {
    loadAvailableModels();
    loadModelInfo();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const response = await imageClassificationAPI.getAvailableModels();
      setModels(response.models || []);
    } catch (error) {
      console.error('Failed to load available models:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadModelInfo = async () => {
    try {
      const info = await imageClassificationAPI.getModelInfo();
      setModelInfo(info);
    } catch (error) {
      console.error('Failed to load model info:', error);
    }
  };

  const handleModelSwitch = async (modelName) => {
    if (modelName === selectedModel || switching) return;

    setSwitching(true);
    try {
      const response = await imageClassificationAPI.switchModel(modelName);
      onModelChange(modelName);
      setModelInfo(response.model_info);
    } catch (error) {
      console.error('Failed to switch model:', error);
      // Show error toast or notification here
    } finally {
      setSwitching(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-2">
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Settings className="w-5 h-5 text-primary-600" />
        <h3 className="text-lg font-semibold text-gray-900">Model Configuration</h3>
      </div>

      {/* Model Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Model
        </label>
        <select
          value={selectedModel}
          onChange={(e) => handleModelSwitch(e.target.value)}
          disabled={isLoading || switching}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.name} - {model.description}
            </option>
          ))}
        </select>
        {switching && (
          <p className="text-sm text-primary-600 mt-2">Switching model...</p>
        )}
      </div>

      {/* Model Information */}
      {modelInfo && (
        <div className="space-y-4">
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Current Model Information</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Name</p>
                <p className="font-medium text-gray-900">{modelInfo.name}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Version</p>
                <p className="font-medium text-gray-900">{modelInfo.version}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Input Size</p>
                <p className="font-medium text-gray-900">
                  {modelInfo.input_size[0]} × {modelInfo.input_size[1]}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Classes</p>
                <p className="font-medium text-gray-900">{modelInfo.num_classes}</p>
              </div>
            </div>
          </div>

          <div>
            <p className="text-sm text-gray-600 mb-2">Description</p>
            <p className="text-sm text-gray-900">{modelInfo.description}</p>
          </div>

          {/* Model Status */}
          <div className="flex items-center space-x-2 pt-2 border-t">
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span className="text-sm text-green-700">Model loaded and ready</span>
          </div>
        </div>
      )}

      {/* Model Comparison */}
      <div className="mt-6 pt-4 border-t">
        <h4 className="text-sm font-medium text-gray-900 mb-3">Available Models</h4>
        <div className="space-y-2">
          {models.map((model) => (
            <div
              key={model.name}
              className={`p-3 rounded-lg border ${
                model.name === selectedModel
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-200 hover:bg-gray-50'
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">{model.name}</p>
                  <p className="text-sm text-gray-600">{model.description}</p>
                </div>
                {model.name === selectedModel && (
                  <Brain className="w-5 h-5 text-primary-600" />
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelSelector;
