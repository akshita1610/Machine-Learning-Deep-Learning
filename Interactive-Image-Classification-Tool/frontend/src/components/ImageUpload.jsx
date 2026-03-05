import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Image as ImageIcon } from 'lucide-react';
import { validateImageFile, formatFileSize, truncateFileName, createImagePreview } from '../utils/helpers';

const ImageUpload = ({ onFilesChange, maxFiles = 10 }) => {
  const [files, setFiles] = useState([]);
  const [errors, setErrors] = useState([]);

  const onDrop = useCallback(async (acceptedFiles, fileRejections) => {
    // Clear previous errors
    setErrors([]);
    
    // Process file rejections
    const newErrors = fileRejections.map(({ file, errors }) => ({
      file: file.name,
      error: errors[0]?.message || 'Invalid file'
    }));
    
    // Validate accepted files
    const validFiles = [];
    const validationErrors = [];
    
    for (const file of acceptedFiles) {
      const validation = validateImageFile(file);
      if (validation.valid) {
        validFiles.push(file);
      } else {
        validationErrors.push({
          file: file.name,
          error: validation.error
        });
      }
    }
    
    setErrors([...newErrors, ...validationErrors]);
    
    // Create preview URLs for valid files
    const filesWithPreviews = await Promise.all(
      validFiles.map(async (file) => ({
        file,
        preview: await createImagePreview(file),
        id: Math.random().toString(36).substr(2, 9)
      }))
    );
    
    const updatedFiles = [...files, ...filesWithPreviews].slice(0, maxFiles);
    setFiles(updatedFiles);
    onFilesChange(updatedFiles.map(f => f.file));
  }, [files, maxFiles, onFilesChange]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.webp']
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: true,
    maxFiles
  });

  const removeFile = (fileId) => {
    const updatedFiles = files.filter(f => f.id !== fileId);
    setFiles(updatedFiles);
    onFilesChange(updatedFiles.map(f => f.file));
  };

  return (
    <div className="w-full">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive 
            ? 'border-primary-500 bg-primary-50' 
            : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          }
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center space-y-4">
          <div className="p-4 bg-primary-100 rounded-full">
            <Upload className="w-8 h-8 text-primary-600" />
          </div>
          <div>
            <p className="text-lg font-medium text-gray-900">
              {isDragActive ? 'Drop the images here...' : 'Drag & drop images here'}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              or click to select files
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Supports: JPEG, PNG, BMP, TIFF, WebP (Max: 10MB each, Max {maxFiles} files)
            </p>
          </div>
        </div>
      </div>

      {/* Errors */}
      {errors.length > 0 && (
        <div className="mt-4 space-y-2">
          {errors.map((error, index) => (
            <div key={index} className="bg-red-50 border border-red-200 rounded-md p-3">
              <p className="text-sm text-red-800">
                <span className="font-medium">{error.file}:</span> {error.error}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* File Preview */}
      {files.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Uploaded Images ({files.length}/{maxFiles})
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {files.map((fileObj) => (
              <div key={fileObj.id} className="relative group">
                <div className="aspect-square rounded-lg overflow-hidden bg-gray-100 border border-gray-200">
                  <img
                    src={fileObj.preview}
                    alt={fileObj.file.name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <button
                  onClick={() => removeFile(fileObj.id)}
                  className="absolute -top-2 -right-2 p-1 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X className="w-4 h-4" />
                </button>
                <div className="mt-2">
                  <p className="text-sm font-medium text-gray-900 truncate" title={fileObj.file.name}>
                    {truncateFileName(fileObj.file.name)}
                  </p>
                  <p className="text-xs text-gray-500">
                    {formatFileSize(fileObj.file.size)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
