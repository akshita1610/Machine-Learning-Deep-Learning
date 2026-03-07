import React from 'react';
import { CheckCircle, XCircle, Download, Brain } from 'lucide-react';
import { formatConfidence } from '../utils/helpers';

const ClassificationResults = ({ results, processingTime, modelUsed, onDownload }) => {
  const successfulResults = results.filter(r => r.success);
  const failedResults = results.filter(r => !r.success);

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBgColor = (confidence) => {
    if (confidence >= 0.8) return 'bg-green-100';
    if (confidence >= 0.6) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  return (
    <div className="w-full space-y-6 animate-fade-in">
      {/* Summary */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Classification Results</h3>
          <div className="flex items-center space-x-2">
            <Brain className="w-5 h-5 text-primary-600" />
            <span className="text-sm text-gray-600">{modelUsed}</span>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-blue-900">{results.length}</div>
            <div className="text-sm text-blue-700">Total Images</div>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-green-900">{successfulResults.length}</div>
            <div className="text-sm text-green-700">Successful</div>
          </div>
          <div className="bg-red-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-red-900">{failedResults.length}</div>
            <div className="text-sm text-red-700">Failed</div>
          </div>
          <div className="bg-purple-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-purple-900">{processingTime.toFixed(2)}s</div>
            <div className="text-sm text-purple-700">Processing Time</div>
          </div>
        </div>

        <div className="flex justify-end">
          <button
            onClick={() => onDownload('json')}
            className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            <Download className="w-4 h-4" />
            <span>Download JSON</span>
          </button>
        </div>
      </div>

      {/* Successful Results */}
      {successfulResults.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">Successful Classifications</h4>
          <div className="space-y-3">
            {successfulResults.map((result, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                    <div>
                      <p className="font-medium text-gray-900">{result.filename}</p>
                      <p className="text-sm text-gray-600">Class: <span className="font-medium">{result.class_name}</span></p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getConfidenceBgColor(result.confidence)} ${getConfidenceColor(result.confidence)}`}>
                      {formatConfidence(result.confidence)}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{result.processing_time.toFixed(3)}s</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Failed Results */}
      {failedResults.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">Failed Classifications</h4>
          <div className="space-y-3">
            {failedResults.map((result, index) => (
              <div key={index} className="border border-red-200 rounded-lg p-4 bg-red-50">
                <div className="flex items-center space-x-3">
                  <XCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">{result.filename}</p>
                    <p className="text-sm text-red-600">{result.error}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Class Distribution */}
      {successfulResults.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">Class Distribution</h4>
          <div className="space-y-2">
            {Object.entries(
              successfulResults.reduce((acc, result) => {
                acc[result.class_name] = (acc[result.class_name] || 0) + 1;
                return acc;
              }, {})
            ).map(([className, count]) => (
              <div key={className} className="flex items-center justify-between">
                <span className="text-sm text-gray-700">{className}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full"
                      style={{ width: `${(count / successfulResults.length) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-600 w-8">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ClassificationResults;
