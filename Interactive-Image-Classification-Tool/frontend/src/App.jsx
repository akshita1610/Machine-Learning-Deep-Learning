import React, { useState } from 'react';
import { Brain, Play, Loader2 } from 'lucide-react';
import ImageUpload from './ImageUpload';
import ClassificationResults from './ClassificationResults';
import ModelSelector from './ModelSelector';
import { imageClassificationAPI } from '../services/api';
import { downloadResults } from '../utils/helpers';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const App = () => {
  const [files, setFiles] = useState([]);
  const [selectedModel, setSelectedModel] = useState('resnet50');
  const [results, setResults] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [processingTime, setProcessingTime] = useState(0);
  const [modelUsed, setModelUsed] = useState('');

  const handleFilesChange = (newFiles) => {
    setFiles(newFiles);
    setResults(null); // Clear previous results when files change
  };

  const handleClassify = async () => {
    if (files.length === 0) {
      toast.error('Please upload at least one image to classify');
      return;
    }

    setIsClassifying(true);
    setResults(null);

    try {
      const startTime = Date.now();
      const response = await imageClassificationAPI.classifyImages(files, selectedModel);
      const endTime = Date.now();

      setResults(response.results);
      setProcessingTime((endTime - startTime) / 1000);
      setModelUsed(response.model_used);

      // Show success toast
      const successful = response.results.filter(r => r.success).length;
      const failed = response.results.filter(r => !r.success).length;
      
      if (failed === 0) {
        toast.success(`Successfully classified all ${successful} images!`);
      } else {
        toast.warning(`Classified ${successful} images successfully. ${failed} failed.`);
      }

    } catch (error) {
      console.error('Classification error:', error);
      toast.error(error.response?.data?.detail || 'Classification failed. Please try again.');
    } finally {
      setIsClassifying(false);
    }
  };

  const handleDownload = (format) => {
    if (!results) return;

    try {
      downloadResults(results, format);
      toast.success(`Results downloaded as ${format.toUpperCase()}`);
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Failed to download results');
    }
  };

  const handleModelChange = (modelName) => {
    setSelectedModel(modelName);
    setResults(null); // Clear results when model changes
    toast.success(`Switched to ${modelName} model`);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Interactive Image Classification Tool
                </h1>
                <p className="text-sm text-gray-600">
                  Upload images and classify them using state-of-the-art ML models
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Upload and Model Selection */}
          <div className="lg:col-span-1 space-y-8">
            {/* Model Selector */}
            <ModelSelector
              selectedModel={selectedModel}
              onModelChange={handleModelChange}
              isLoading={isClassifying}
            />

            {/* Image Upload */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Images</h2>
              <ImageUpload
                onFilesChange={handleFilesChange}
                maxFiles={10}
              />
            </div>

            {/* Classify Button */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <button
                onClick={handleClassify}
                disabled={files.length === 0 || isClassifying}
                className="w-full flex items-center justify-center space-x-2 px-6 py-3 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isClassifying ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Classifying...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Classify Images</span>
                  </>
                )}
              </button>
              
              {files.length > 0 && !isClassifying && (
                <p className="text-sm text-gray-600 mt-2 text-center">
                  Ready to classify {files.length} image{files.length > 1 ? 's' : ''}
                </p>
              )}
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-2">
            {results ? (
              <ClassificationResults
                results={results}
                processingTime={processingTime}
                modelUsed={modelUsed}
                onDownload={handleDownload}
              />
            ) : (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
                <div className="max-w-md mx-auto">
                  <div className="p-4 bg-gray-100 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                    <Brain className="w-8 h-8 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    No Classification Results Yet
                  </h3>
                  <p className="text-gray-600 mb-6">
                    Upload images and click "Classify Images" to see the results here.
                  </p>
                  <div className="text-sm text-gray-500 space-y-1">
                    <p>• Upload up to 10 images at once</p>
                    <p>• Choose from multiple ML models</p>
                    <p>• Get confidence scores and class predictions</p>
                    <p>• Download results in JSON or CSV format</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-500">
            <p>Interactive Image Classification Tool - Powered by Deep Learning</p>
            <p className="mt-1">Built with React, FastAPI, and PyTorch</p>
          </div>
        </div>
      </footer>

      {/* Toast Container */}
      <ToastContainer
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
      />
    </div>
  );
};

export default App;
