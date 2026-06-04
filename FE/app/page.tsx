'use client';

import { useState, useCallback } from 'react';
import { CircleAlert as AlertCircle, X, CircleCheck as CheckCircle2 } from 'lucide-react';
import Header from '@/components/demo/Header';
import ImageUpload from '@/components/demo/ImageUpload';
import ControlPanel from '@/components/demo/ControlPanel';
import ResultGrid from '@/components/demo/ResultGrid';
import MetricsSection from '@/components/demo/MetricsSection';
import ModelExplanation from '@/components/demo/ModelExplanation';
import { PredictionResponse } from '@/types/demo';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.55);
  const [usePostprocess, setUsePostprocess] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const handleImageSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  }, []);

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setResult(null);
    setError(null);
  }, []);

  const handleRunPrediction = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('threshold', threshold.toString());
    formData.append('use_postprocess', usePostprocess.toString());

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch {
      setError('Prediction failed. Please check that the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-sky-50/20">
      <Header />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

        {/* Error Banner */}
        {error && (
          <div className="flex items-start gap-3 bg-red-50 border border-red-200 text-red-700 rounded-2xl px-5 py-4">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <p className="text-sm flex-1">{error}</p>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-600 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Success Banner */}
        {result && !error && (
          <div className="flex items-center gap-3 bg-green-50 border border-green-200 text-green-700 rounded-2xl px-5 py-4">
            <CheckCircle2 className="w-5 h-5 shrink-0" />
            <p className="text-sm font-medium">
              Prediction complete. Manipulated region analysis ready.
            </p>
          </div>
        )}

        {/* Upload + Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Panel */}
          <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-4">
            <div>
              <h2 className="text-lg font-bold text-slate-800 mb-1">Upload Image</h2>
              <p className="text-sm text-slate-400">
                Select or drag &amp; drop an image to analyze for manipulation.
              </p>
            </div>
            <ImageUpload
              onImageSelect={handleImageSelect}
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              onClear={handleClear}
            />
          </div>

          {/* Controls Panel */}
          <div className="space-y-4">
            <div className="bg-gradient-to-br from-blue-50 to-sky-50 rounded-2xl border border-blue-100 p-5">
              <h2 className="text-lg font-bold text-slate-800 mb-1">Configuration</h2>
              <p className="text-sm text-slate-500">
                Adjust prediction parameters and run inference.
              </p>
            </div>
            <ControlPanel
              threshold={threshold}
              onThresholdChange={setThreshold}
              usePostprocess={usePostprocess}
              onPostprocessChange={setUsePostprocess}
              onRunPrediction={handleRunPrediction}
              isLoading={isLoading}
              disabled={!selectedFile}
            />
          </div>
        </div>

        {/* Results Section */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-slate-800">Prediction Results</h2>
              <p className="text-sm text-slate-400 mt-0.5">
                Four-panel visualization of the detection pipeline output.
              </p>
            </div>
            {result && (
              <span className="text-xs font-medium text-green-700 bg-green-50 border border-green-200 px-3 py-1 rounded-full">
                Analysis Complete
              </span>
            )}
          </div>

          <ResultGrid
            originalUrl={previewUrl}
            probabilityMap={result?.probability_map ?? null}
            mask={result?.mask ?? null}
            overlay={result?.overlay ?? null}
          />
        </section>

        {/* Metrics Section */}
        <section className="space-y-4">
          <div>
            <h2 className="text-xl font-bold text-slate-800">Detection Metrics</h2>
            <p className="text-sm text-slate-400 mt-0.5">
              Quantitative statistics from the probability map analysis.
            </p>
          </div>
          <MetricsSection
            metrics={
              result
                ? {
                    raw_prob_min: result.raw_prob_min,
                    raw_prob_mean: result.raw_prob_mean,
                    raw_prob_max: result.raw_prob_max,
                    pred_area_ratio: result.pred_area_ratio,
                    threshold: result.threshold,
                  }
                : null
            }
          />
        </section>

        {/* Model Explanation */}
        <ModelExplanation />
      </main>

      <footer className="mt-16 border-t border-slate-100 bg-white/60 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-blue-500" />
            <span className="text-sm font-medium text-slate-600">
              Computer Vision Research Demo
            </span>
          </div>
          <div className="flex items-center gap-6">
            <span className="text-xs text-slate-400">U-Net++ · ResNet34 · CBAM</span>
            <span className="text-xs text-slate-300">|</span>
            <span className="text-xs text-slate-400">
              API:{' '}
              <code className="text-blue-500 font-mono">{API_URL}</code>
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
