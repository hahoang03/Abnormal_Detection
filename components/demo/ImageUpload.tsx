'use client';

import { useRef, useState, useCallback } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  selectedFile: File | null;
  previewUrl: string | null;
  onClear: () => void;
}

export default function ImageUpload({
  onImageSelect,
  selectedFile,
  previewUrl,
  onClear,
}: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (file && file.type.startsWith('image/')) {
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-4">
      <div
        onClick={() => !selectedFile && inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative border-2 border-dashed rounded-2xl transition-all duration-200
          ${selectedFile
            ? 'border-blue-300 bg-blue-50/30'
            : isDragging
            ? 'border-blue-500 bg-blue-50 scale-[1.01]'
            : 'border-blue-200 bg-slate-50/50 hover:border-blue-400 hover:bg-blue-50/50 cursor-pointer'
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleInputChange}
        />

        {previewUrl ? (
          <div className="relative p-3">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
              className="absolute top-5 right-5 z-10 p-1.5 bg-white/90 hover:bg-red-50 border border-slate-200 hover:border-red-300 text-slate-500 hover:text-red-500 rounded-full shadow-sm transition-all"
            >
              <X className="w-4 h-4" />
            </button>
            <img
              src={previewUrl}
              alt="Preview"
              className="w-full max-h-72 object-contain rounded-xl"
            />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
            <div
              className={`p-4 rounded-2xl mb-4 transition-colors ${
                isDragging ? 'bg-blue-100' : 'bg-blue-50'
              }`}
            >
              <Upload
                className={`w-8 h-8 transition-colors ${
                  isDragging ? 'text-blue-600' : 'text-blue-400'
                }`}
              />
            </div>
            <p className="text-slate-700 font-medium mb-1">
              {isDragging ? 'Release to upload' : 'Drag & drop an image here'}
            </p>
            <p className="text-slate-400 text-sm mb-4">or click to browse files</p>
            <span className="text-xs text-slate-400 bg-slate-100 px-3 py-1 rounded-full">
              PNG, JPG, JPEG, WEBP supported
            </span>
          </div>
        )}
      </div>

      {selectedFile && (
        <div className="flex items-center gap-3 bg-blue-50 border border-blue-100 rounded-xl px-4 py-3">
          <div className="p-2 bg-blue-100 rounded-lg">
            <ImageIcon className="w-4 h-4 text-blue-600" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-slate-700 truncate">{selectedFile.name}</p>
            <p className="text-xs text-slate-400">{formatFileSize(selectedFile.size)}</p>
          </div>
          <span className="text-xs font-medium text-blue-600 bg-blue-100 px-2 py-0.5 rounded-full shrink-0">
            Ready
          </span>
        </div>
      )}
    </div>
  );
}
