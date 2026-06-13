'use client';

import { Play, Loader as Loader2, SlidersHorizontal, Zap, Brain } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';

type PredictionModel = 'hcunetpp' | 'unetpp' | 'compare';

interface ControlPanelProps {
  threshold: number;
  onThresholdChange: (value: number) => void;
  usePostprocess: boolean;
  onPostprocessChange: (value: boolean) => void;
  selectedModel: PredictionModel;
  onModelChange: (value: PredictionModel) => void;
  onRunPrediction: () => void;
  isLoading: boolean;
  disabled: boolean;
}

export default function ControlPanel({
  threshold,
  onThresholdChange,
  usePostprocess,
  onPostprocessChange,
  selectedModel,
  onModelChange,
  onRunPrediction,
  isLoading,
  disabled,
}: ControlPanelProps) {
  const getThresholdLabel = (val: number) => {
    if (val < 0.2) return 'Very Sensitive';
    if (val < 0.4) return 'Sensitive';
    if (val < 0.6) return 'Balanced';
    if (val < 0.75) return 'Conservative';
    return 'Very Conservative';
  };

  const getThresholdColor = (val: number) => {
    if (val < 0.2) return 'text-rose-500';
    if (val < 0.4) return 'text-amber-500';
    if (val < 0.6) return 'text-blue-600';
    if (val < 0.75) return 'text-teal-600';
    return 'text-slate-500';
  };

  const modelOptions: {
    value: PredictionModel;
    title: string;
    description: string;
  }[] = [
    {
      value: 'hcunetpp',
      title: 'HCUNet++',
      description: 'Proposed model',
    },
    {
      value: 'unetpp',
      title: 'U-Net++',
      description: 'Baseline model',
    },
    {
      value: 'compare',
      title: 'Compare Both',
      description: 'Run both models',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-6">
        <div className="flex items-center gap-2 mb-1">
          <div className="p-1.5 bg-blue-50 rounded-lg">
            <SlidersHorizontal className="w-4 h-4 text-blue-600" />
          </div>
          <h3 className="font-semibold text-slate-800">Detection Parameters</h3>
        </div>

        {/* Threshold Slider */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium text-slate-600">
              Detection Threshold
            </Label>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold text-blue-600 tabular-nums">
                {threshold.toFixed(2)}
              </span>
              <span
                className={`text-xs font-medium px-2 py-0.5 rounded-full bg-slate-100 ${getThresholdColor(
                  threshold
                )}`}
              >
                {getThresholdLabel(threshold)}
              </span>
            </div>
          </div>

          <Slider
            min={0.05}
            max={0.85}
            step={0.01}
            value={[threshold]}
            onValueChange={([val]) => onThresholdChange(val)}
            className="w-full"
          />

          <div className="flex justify-between text-xs text-slate-400">
            <span>0.05 (More detections)</span>
            <span>0.85 (Fewer detections)</span>
          </div>

          <p className="text-xs text-slate-400 bg-slate-50 rounded-lg px-3 py-2">
            Pixels with probability above this threshold are classified as manipulated.
          </p>
        </div>

        {/* Post-processing Toggle */}
        <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
          <div className="flex items-center gap-3">
            <div className="p-1.5 bg-sky-100 rounded-lg">
              <Zap className="w-4 h-4 text-sky-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700">Post-processing</p>
              <p className="text-xs text-slate-400">Morphological refinement</p>
            </div>
          </div>

          <Switch
            checked={usePostprocess}
            onCheckedChange={onPostprocessChange}
          />
        </div>

        {/* Prediction Model Selector */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-indigo-50 rounded-lg">
              <Brain className="w-4 h-4 text-indigo-600" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-700">Prediction Model</p>
              <p className="text-xs text-slate-400">
                Select one model or compare HCUNet++ with U-Net++.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {modelOptions.map((option) => {
              const isSelected = selectedModel === option.value;

              return (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => onModelChange(option.value)}
                  className={`text-left rounded-xl border px-4 py-3 transition-all duration-200 ${
                    isSelected
                      ? 'border-blue-400 bg-blue-50 shadow-sm'
                      : 'border-slate-100 bg-slate-50 hover:border-blue-200 hover:bg-blue-50/50'
                  }`}
                >
                  <p
                    className={`text-sm font-semibold ${
                      isSelected ? 'text-blue-700' : 'text-slate-700'
                    }`}
                  >
                    {option.title}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    {option.description}
                  </p>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Run Button */}
      <Button
        onClick={onRunPrediction}
        disabled={disabled || isLoading}
        className="w-full h-12 text-base font-semibold bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white rounded-xl shadow-md hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? (
          <span className="flex items-center gap-2">
            <Loader2 className="w-5 h-5 animate-spin" />
            {selectedModel === 'compare' ? 'Comparing Models...' : 'Analyzing Image...'}
          </span>
        ) : (
          <span className="flex items-center gap-2">
            <Play className="w-5 h-5" />
            {selectedModel === 'compare' ? 'Run Comparison' : 'Run Prediction'}
          </span>
        )}
      </Button>

      {disabled && !isLoading && (
        <p className="text-center text-xs text-slate-400">
          Upload an image to enable prediction
        </p>
      )}
    </div>
  );
}