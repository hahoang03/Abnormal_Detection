'use client';

import { useState, useCallback, useEffect, type DragEvent } from 'react';
import {
  CircleAlert as AlertCircle,
  X,
  CircleCheck as CheckCircle2,
} from 'lucide-react';

import Header from '@/components/demo/Header';
import ImageUpload from '@/components/demo/ImageUpload';
import ControlPanel from '@/components/demo/ControlPanel';
import ResultGrid from '@/components/demo/ResultGrid';
import MetricsSection from '@/components/demo/MetricsSection';
import ModelExplanation from '@/components/demo/ModelExplanation';
import type { PredictionResponse } from '@/types/demo';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const GENERATE_API_URL =
  process.env.NEXT_PUBLIC_GENERATE_API_URL || 'http://localhost:8001';

type EditMode = 'clean' | 'shape_overlay' | 'sd_inpaint';

type PredictionInputSource = 'generated' | 'upload';

type PredictionModel = 'hcunetpp' | 'unetpp' | 'compare';

type ComparisonResult = {
  hcunetpp: PredictionResponse | null;
  unetpp: PredictionResponse | null;
};

type ComparisonMetrics = {
  hcunetpp: DemoMetrics | null;
  unetpp: DemoMetrics | null;
};

type MaskSizeLevel = 'random' | 'small' | 'medium' | 'large';

type InpaintShape =
  | 'random'
  | 'ellipse'
  | 'circle'
  | 'rounded_rect'
  | 'irregular';

type OverlayShape =
  | 'random'
  | 'rect'
  | 'rounded_rect'
  | 'circle'
  | 'ellipse';

type GenerateResponse = {
  original?: string;
  edited?: string;
  mask?: string;
  overlay?: string;

  generated_image?: string;
  edited_image?: string;
  image?: string;
  gt_mask?: string;

  error?: string;
  info?: {
    edit_group?: string;
    edit_type?: string;
    status?: string;
    shape?: string;
    level?: string;
    area_ratio?: number;
    prompt?: string;
    cue_mode?: string;
    strength?: number;
    guidance_scale?: number;
    seed?: number;
    retry_count?: number;
    diff_mean?: number;
    diff_median?: number;
    diff_p75?: number;
    diff_p90?: number;
    changed_ratio_5?: number;
    changed_ratio_10?: number;
    changed_ratio_20?: number;
    opacity?: number;
    edge_blur?: number;
    overlay_color?: number[];
  };
};

type DemoMetrics = {
  dice: number;
  iou: number;
  gt_area: number;
  pred_area: number;
  gt_pixels: number;
  pred_pixels: number;
  threshold: number;
};

function looksLikeRawBase64Image(value: string) {
  const compact = value.replace(/\s/g, '');

  if (compact.length < 200) return false;

  if (
    compact.startsWith('/9j/') ||
    compact.startsWith('iVBOR') ||
    compact.startsWith('R0lGOD') ||
    compact.startsWith('UklGR')
  ) {
    return true;
  }

  return /^[A-Za-z0-9+/=]+$/.test(compact);
}

function rawBase64ToDataUrl(value: string) {
  const compact = value.replace(/\s/g, '');

  let mimeType = 'image/png';

  if (compact.startsWith('/9j/')) {
    mimeType = 'image/jpeg';
  } else if (compact.startsWith('R0lGOD')) {
    mimeType = 'image/gif';
  } else if (compact.startsWith('UklGR')) {
    mimeType = 'image/webp';
  }

  return `data:${mimeType};base64,${compact}`;
}

function resolveImageSrc(url?: string | null, baseUrl = '') {
  if (!url) return null;

  const trimmedUrl = url.trim();

  if (
    trimmedUrl.startsWith('http://') ||
    trimmedUrl.startsWith('https://') ||
    trimmedUrl.startsWith('data:') ||
    trimmedUrl.startsWith('blob:')
  ) {
    return trimmedUrl;
  }

  if (looksLikeRawBase64Image(trimmedUrl)) {
    return rawBase64ToDataUrl(trimmedUrl);
  }

  return `${baseUrl}${trimmedUrl.startsWith('/') ? '' : '/'}${trimmedUrl}`;
}

function resolveGenerateUrl(url?: string | null) {
  return resolveImageSrc(url, GENERATE_API_URL);
}

function resolvePredictionUrl(url?: string | null) {
  return resolveImageSrc(url, API_URL);
}

async function imageSrcToFile(src: string, filename: string) {
  try {
    const response = await fetch(src);

    if (!response.ok) {
      throw new Error(`Cannot fetch image. Status: ${response.status}`);
    }

    const blob = await response.blob();

    return new File([blob], filename, {
      type: blob.type || 'image/png',
    });
  } catch (error) {
    console.error('Convert image URL to File failed:', error);

    throw new Error(
      'Cannot load generated image for prediction. Please check image URL, CORS, or backend response format.'
    );
  }
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.crossOrigin = 'anonymous';

    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Cannot load mask image.'));

    img.src = src;
  });
}

async function maskToBinaryArray(
  src: string,
  targetWidth?: number,
  targetHeight?: number
) {
  const img = await loadImageElement(src);

  const width = targetWidth ?? img.naturalWidth;
  const height = targetHeight ?? img.naturalHeight;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Cannot create canvas context.');
  }

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height).data;

  const binary = new Uint8Array(width * height);
  let count = 0;

  for (let i = 0; i < width * height; i += 1) {
    const r = imageData[i * 4];
    const g = imageData[i * 4 + 1];
    const b = imageData[i * 4 + 2];

    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

    // Giống Colab: mask pixel > 0 thì xem là vùng được đánh dấu
    if (gray > 0) {
      binary[i] = 1;
      count += 1;
    }
  }

  return {
    binary,
    width,
    height,
    count,
  };
}

async function computeDiceIoUMetrics({
  predMaskUrl,
  gtMaskUrl,
  threshold,
}: {
  predMaskUrl: string;
  gtMaskUrl: string;
  threshold: number;
}): Promise<DemoMetrics> {
  const pred = await maskToBinaryArray(predMaskUrl);
  const gt = await maskToBinaryArray(gtMaskUrl, pred.width, pred.height);

  let intersection = 0;

  for (let i = 0; i < pred.binary.length; i += 1) {
    if (pred.binary[i] === 1 && gt.binary[i] === 1) {
      intersection += 1;
    }
  }

  const eps = 1e-6;

  const predPixels = pred.count;
  const gtPixels = gt.count;

  const dice =
    (2 * intersection + eps) / (predPixels + gtPixels + eps);

  const iou =
    (intersection + eps) /
    (predPixels + gtPixels - intersection + eps);

  const totalPixels = pred.width * pred.height;

  return {
    dice,
    iou,
    gt_area: gtPixels / totalPixels,
    pred_area: predPixels / totalPixels,
    gt_pixels: gtPixels,
    pred_pixels: predPixels,
    threshold,
  };
}

export default function Home() {
  const [activeStep, setActiveStep] = useState<1 | 2 | 3 | 4>(1);

  const [originalFile, setOriginalFile] = useState<File | null>(null);
  const [originalPreviewUrl, setOriginalPreviewUrl] = useState<string | null>(
    null
  );

  const [editMode, setEditMode] = useState<EditMode>('sd_inpaint');
  const applySubtleCue = true;

  const [predictionSource, setPredictionSource] =
    useState<PredictionInputSource>('generated');

  const [predictionInputFile, setPredictionInputFile] = useState<File | null>(null);

  const [predictionInputPreviewUrl, setPredictionInputPreviewUrl] = useState<string | null>(null);
  const [isPredictionDragOver, setIsPredictionDragOver] = useState(false);


  const [maskSizeLevel, setMaskSizeLevel] =
    useState<MaskSizeLevel>('random');

  const [inpaintShape, setInpaintShape] =
    useState<InpaintShape>('random');

  const [overlayShape, setOverlayShape] =
    useState<OverlayShape>('random');

  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(
    null
  );
  const [gtMaskUrl, setGtMaskUrl] = useState<string | null>(null);
  const [generatedOverlayUrl, setGeneratedOverlayUrl] = useState<string | null>(
    null
  );

  const predictionInputUrl =
    predictionSource === 'upload'
      ? predictionInputPreviewUrl
      : generatedImageUrl;


  const [manualGtMaskUrl, setManualGtMaskUrl] = useState<string | null>(null);

  const [threshold, setThreshold] = useState(0.55);
  const [usePostprocess, setUsePostprocess] = useState(true);

  const [isGenerating, setIsGenerating] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [demoMetrics, setDemoMetrics] = useState<DemoMetrics | null>(null);

  const [selectedModel, setSelectedModel] =
    useState<PredictionModel>('hcunetpp');

  const [comparisonResult, setComparisonResult] =
    useState<ComparisonResult | null>(null);

  const [comparisonMetrics, setComparisonMetrics] =
    useState<ComparisonMetrics>({
      hcunetpp: null,
      unetpp: null,
    });

  const activeGtMaskUrl = manualGtMaskUrl || gtMaskUrl;

  useEffect(() => {
    const predMaskUrl = result?.mask;

    if (!activeGtMaskUrl || !predMaskUrl) {
      setDemoMetrics(null);
      return;
    }

    let cancelled = false;

    const calculateMetrics = async () => {
      try {
        const calculatedMetrics = await computeDiceIoUMetrics({
          predMaskUrl,
          gtMaskUrl: activeGtMaskUrl,
          threshold: result?.threshold ?? threshold,
        });

        if (!cancelled) {
          setDemoMetrics(calculatedMetrics);
        }
      } catch (metricError) {
        console.error('Metric calculation error:', metricError);

        if (!cancelled) {
          setDemoMetrics(null);
          setError(
            'Dice/IoU metrics could not be calculated. Please check GT mask and predicted mask format.'
          );
        }
      }
    };

    calculateMetrics();

    return () => {
      cancelled = true;
    };
  }, [activeGtMaskUrl, result?.mask, result?.threshold, threshold]);

  useEffect(() => {
    if (!activeGtMaskUrl || !comparisonResult) {
      setComparisonMetrics({
        hcunetpp: null,
        unetpp: null,
      });
      return;
    }

    let cancelled = false;

    const calculateComparisonMetrics = async () => {
      try {
        const hcunetppMetrics = comparisonResult.hcunetpp?.mask
          ? await computeDiceIoUMetrics({
            predMaskUrl: comparisonResult.hcunetpp.mask,
            gtMaskUrl: activeGtMaskUrl,
            threshold: comparisonResult.hcunetpp.threshold ?? threshold,
          })
          : null;

        const unetppMetrics = comparisonResult.unetpp?.mask
          ? await computeDiceIoUMetrics({
            predMaskUrl: comparisonResult.unetpp.mask,
            gtMaskUrl: activeGtMaskUrl,
            threshold: comparisonResult.unetpp.threshold ?? threshold,
          })
          : null;

        if (!cancelled) {
          setComparisonMetrics({
            hcunetpp: hcunetppMetrics,
            unetpp: unetppMetrics,
          });
        }
      } catch (metricError) {
        console.error('Comparison metric calculation error:', metricError);

        if (!cancelled) {
          setComparisonMetrics({
            hcunetpp: null,
            unetpp: null,
          });
        }
      }
    };

    calculateComparisonMetrics();

    return () => {
      cancelled = true;
    };
  }, [
    activeGtMaskUrl,
    comparisonResult?.hcunetpp?.mask,
    comparisonResult?.hcunetpp?.threshold,
    comparisonResult?.unetpp?.mask,
    comparisonResult?.unetpp?.threshold,
    threshold,
  ]);


  const handleImageSelect = useCallback((file: File) => {
    setOriginalFile(file);

    setGeneratedImageUrl(null);
    setGtMaskUrl(null);
    setGeneratedOverlayUrl(null);
    setManualGtMaskUrl(null);
    setDemoMetrics(null);

    setResult(null);
    setComparisonResult(null);
    setComparisonMetrics({
      hcunetpp: null,
      unetpp: null,
    });
    setError(null);
    setSuccessMessage(null);

    const url = URL.createObjectURL(file);

    setOriginalPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  }, []);

  const handleClear = useCallback(() => {
    setOriginalFile(null);

    setOriginalPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });

    setGeneratedImageUrl(null);
    setGtMaskUrl(null);
    setGeneratedOverlayUrl(null);
    setDemoMetrics(null);

    setManualGtMaskUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });

    setResult(null);
    setComparisonResult(null);
    setComparisonMetrics({
      hcunetpp: null,
      unetpp: null,
    });
    setError(null);
    setSuccessMessage(null);
  }, []);

  const handleManualGtMaskSelect = (file: File) => {
    const url = URL.createObjectURL(file);

    setManualGtMaskUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  };

  const handlePredictionImageSelect = (file: File) => {
    const url = URL.createObjectURL(file);

    setPredictionSource('upload');
    setPredictionInputFile(file);

    setPredictionInputPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });

    setResult(null);
    setComparisonResult(null);
    setComparisonMetrics({
      hcunetpp: null,
      unetpp: null,
    });
    setDemoMetrics(null);
    setError(null);
    setSuccessMessage(null);

    setManualGtMaskUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
  };

  const handlePredictionDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsPredictionDragOver(true);
  };

  const handlePredictionDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsPredictionDragOver(false);
  };

  const handlePredictionImageDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();

    setIsPredictionDragOver(false);

    const file = event.dataTransfer.files?.[0];
    if (!file) return;

    handlePredictionImageSelect(file);
  };
  const handleClearPredictionUpload = () => {
    setPredictionInputFile(null);

    setPredictionInputPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });

    setResult(null);
    setComparisonResult(null);
    setComparisonMetrics({
      hcunetpp: null,
      unetpp: null,
    });
    setDemoMetrics(null);
    setError(null);
    setSuccessMessage(null);

    if (generatedImageUrl) {
      setPredictionSource('generated');
    }
  };

  const handleGenerate = async () => {
    if (!originalFile) return;

    setIsGenerating(true);
    setError(null);
    setSuccessMessage(null);
    setDemoMetrics(null);

    setGeneratedImageUrl(null);
    setGtMaskUrl(null);
    setGeneratedOverlayUrl(null);
    setManualGtMaskUrl(null);
    setResult(null);
    setComparisonResult(null);
    setComparisonMetrics({
      hcunetpp: null,
      unetpp: null,
    });

    const formData = new FormData();
    formData.append('file', originalFile);

    const selectedShape =
      editMode === 'sd_inpaint'
        ? inpaintShape
        : editMode === 'shape_overlay'
          ? overlayShape
          : 'none';

    const selectedMaskSize =
      editMode === 'clean' ? 'none' : maskSizeLevel;

    formData.append('mode', editMode);

    formData.append(
      'use_subtle_cue',
      (editMode === 'sd_inpaint').toString()
    );

    // QUAN TRỌNG: backend đọc field này
    formData.append('mask_size', selectedMaskSize);

    // Có thể gửi thêm camelCase để chắc chắn nếu backend có fallback
    formData.append('maskSize', selectedMaskSize);

    // Shape hiện tại bạn gửi đúng rồi
    formData.append('shape', selectedShape);

    // Debug để check FE gửi gì xuống BE
    for (const [key, value] of formData.entries()) {
      console.log('Generate FormData:', key, value);
    }

    try {
      const response = await fetch(`${GENERATE_API_URL}/api/generate`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(
          `Generate server responded with status ${response.status}. ${text}`
        );
      }

      const data: GenerateResponse = await response.json();

      console.log('Generate response:', data);

      if (data.error) {
        throw new Error(data.error);
      }

      const editedUrl =
        resolveGenerateUrl(data.edited) ||
        resolveGenerateUrl(data.generated_image) ||
        resolveGenerateUrl(data.edited_image) ||
        resolveGenerateUrl(data.image);

      const maskUrl =
        resolveGenerateUrl(data.mask) || resolveGenerateUrl(data.gt_mask);

      const overlayUrl = resolveGenerateUrl(data.overlay);

      if (!editedUrl) {
        throw new Error('No edited/generated image returned from backend.');
      }

      setGeneratedImageUrl(editedUrl);
      setGtMaskUrl(maskUrl);
      setGeneratedOverlayUrl(overlayUrl);

      setSuccessMessage(
        'Edited image generated successfully. You can now run prediction.'
      );
    } catch (err) {
      console.error('Generate error:', err);

      setError(
        err instanceof Error
          ? `Generate failed: ${err.message}`
          : 'Generate failed. Please check that the generation backend server is running.'
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunPrediction = async () => {
    if (!predictionInputUrl) {
      setError(
        'Please generate an edited image first or upload a custom image for prediction.'
      );
      return;
    }

    setIsPredicting(true);
    setError(null);
    setSuccessMessage(null);
    setResult(null);
    setDemoMetrics(null);
    setComparisonResult(null);
    setComparisonMetrics({
      hcunetpp: null,
      unetpp: null,
    });

    try {
      const inputFile =
        predictionSource === 'upload'
          ? predictionInputFile
          : await imageSrcToFile(predictionInputUrl, 'generated-image.png');

      if (!inputFile) {
        throw new Error('No prediction input file found.');
      }

      let overlayMode = 'fill';

      if (predictionSource === 'upload') {
        overlayMode = 'contour';
      } else if (editMode === 'shape_overlay') {
        overlayMode = 'contour';
      } else {
        overlayMode = 'fill';
      }

      const runSingleModel = async (
        modelType: 'hcunetpp' | 'unetpp'
      ): Promise<PredictionResponse> => {
        const formData = new FormData();

        formData.append('file', inputFile);
        formData.append('threshold', threshold.toString());
        formData.append('use_postprocess', usePostprocess.toString());
        formData.append('overlay_mode', overlayMode);
        formData.append('model_type', modelType);

        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const text = await response.text().catch(() => '');
          throw new Error(
            `Prediction server responded with status ${response.status}. ${text}`
          );
        }

        const data: PredictionResponse = await response.json();

        const normalizedResult = {
          ...data,
          probability_map:
            resolvePredictionUrl(data.probability_map) ?? data.probability_map,
          mask: resolvePredictionUrl(data.mask) ?? data.mask,
          overlay: resolvePredictionUrl(data.overlay) ?? data.overlay,
        };

        return normalizedResult as PredictionResponse;
      };

      if (selectedModel === 'compare') {
        const hcunetppResult = await runSingleModel('hcunetpp');
        const unetppResult = await runSingleModel('unetpp');

        setComparisonResult({
          hcunetpp: hcunetppResult,
          unetpp: unetppResult,
        });

        // Main result vẫn dùng HCUNet++ để hiển thị ở phần Review Prediction Result
        setResult(hcunetppResult);

        setSuccessMessage(
          'Model comparison complete. HCUNet++ and U-Net results are ready.'
        );
      } else {
        const singleResult = await runSingleModel(selectedModel);

        setResult(singleResult);
        setComparisonResult(null);

        setSuccessMessage('Prediction complete. Dice and IoU metrics are ready.');
      }
    } catch (err) {
      console.error('Prediction error:', err);

      setError(
        err instanceof Error
          ? `Prediction failed: ${err.message}`
          : 'Prediction failed. Please check that the detection backend server is running.'
      );
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-sky-50/20">
      <Header />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-5">
        {/* Page Title */}
        <section className="space-y-3">
          <div className="space-y-2">
            <h2 className="text-xl sm:text-2xl font-bold text-slate-800">
              Guideline
            </h2>

            <p className="text-sm sm:text-base text-slate-500 whitespace-nowrap">
              Upload an original image, generate an edited version, run model prediction, and review the predicted manipulated region.
            </p>
          </div>

          <div className="bg-white border border-slate-100 rounded-2xl shadow-sm p-3">
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <StepButton
                step={1}
                title="Upload"
                description="Original image"
                activeStep={activeStep}
                onClick={() => setActiveStep(1)}
                completed={!!originalPreviewUrl}
              />

              <StepButton
                step={2}
                title="Generate"
                description="Edited image"
                activeStep={activeStep}
                onClick={() => setActiveStep(2)}
                completed={!!generatedImageUrl}
              />

              <StepButton
                step={3}
                title="Predict"
                description="Run model"
                activeStep={activeStep}
                onClick={() => setActiveStep(3)}
                completed={!!result}
              />

              <StepButton
                step={4}
                title="Review"
                description="Result & metrics"
                activeStep={activeStep}
                onClick={() => setActiveStep(4)}
                completed={!!result || !!demoMetrics}
              />
            </div>
          </div>
        </section>





        {/* Step 1 */}
        {activeStep === 1 && (
          <section className="space-y-4">
            <div className="bg-white rounded-2xl border border-slate-100 shadow-sm  space-y-2">
              <ImageUpload
                onImageSelect={handleImageSelect}
                selectedFile={originalFile}
                previewUrl={originalPreviewUrl}
                onClear={handleClear}
              />
            </div>
          </section>
        )}

        {/* Step 2 */}
        {activeStep === 2 && (
          <section className="space-y-4">
            {/* <div>
              <div className="inline-flex items-center rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-600 mb-3">
                Step 2
              </div>

              <h2 className="text-xl font-bold text-slate-800">
                Generate Edited Image
              </h2>

              <p className="text-sm text-slate-400 mt-1">
                Choose the edit mode and generate an abnormal image for testing.
              </p>
            </div> */}

            <div className="space-y-6">
              {/* Generate Settings */}
              <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">
                      Edit Mode
                    </label>

                    <select
                      value={editMode}
                      onChange={(e) => setEditMode(e.target.value as EditMode)}
                      className="w-full rounded-xl border border-slate-200 px-4 py-3 text-slate-700 outline-none focus:ring-2 focus:ring-blue-200"
                    >
                      <option value="clean">Clean</option>
                      <option value="shape_overlay">Shape Overlay</option>
                      <option value="sd_inpaint">SD Inpaint</option>
                    </select>
                  </div>

                  {editMode !== 'clean' && (
                    <div>
                      <label className="block text-sm font-semibold text-slate-700 mb-2">
                        Mask Size
                      </label>

                      <select
                        value={maskSizeLevel}
                        onChange={(e) =>
                          setMaskSizeLevel(e.target.value as MaskSizeLevel)
                        }
                        className="w-full rounded-xl border border-slate-200 px-4 py-3 text-slate-700 outline-none focus:ring-2 focus:ring-blue-200"
                      >
                        <option value="random">Random</option>
                        <option value="small">Small</option>
                        <option value="medium">Medium</option>
                        <option value="large">Large</option>
                      </select>
                    </div>
                  )}

                  {editMode === 'sd_inpaint' && (
                    <div>
                      <label className="block text-sm font-semibold text-slate-700 mb-2">
                        Inpaint Shape
                      </label>

                      <select
                        value={inpaintShape}
                        onChange={(e) =>
                          setInpaintShape(e.target.value as InpaintShape)
                        }
                        className="w-full rounded-xl border border-slate-200 px-4 py-3 text-slate-700 outline-none focus:ring-2 focus:ring-blue-200"
                      >
                        <option value="random">Random</option>
                        <option value="ellipse">Ellipse</option>
                        <option value="circle">Circle</option>
                        <option value="rounded_rect">Rounded Rectangle</option>
                        <option value="irregular">Irregular</option>
                      </select>
                    </div>
                  )}

                  {editMode === 'shape_overlay' && (
                    <div>
                      <label className="block text-sm font-semibold text-slate-700 mb-2">
                        Overlay Shape
                      </label>

                      <select
                        value={overlayShape}
                        onChange={(e) =>
                          setOverlayShape(e.target.value as OverlayShape)
                        }
                        className="w-full rounded-xl border border-slate-200 px-4 py-3 text-slate-700 outline-none focus:ring-2 focus:ring-blue-200"
                      >
                        <option value="random">Random</option>
                        <option value="rect">Rectangle</option>
                        <option value="rounded_rect">Rounded Rectangle</option>
                        <option value="circle">Circle</option>
                        <option value="ellipse">Ellipse</option>
                      </select>
                    </div>
                  )}
                </div>



                {editMode !== 'clean' && (
                  <div className="rounded-xl bg-slate-50 border border-slate-100 px-4 py-3">
                    <p className="text-xs font-semibold text-slate-500 mb-1">
                      Current Generate Config
                    </p>

                    <p className="text-sm text-slate-700">
                      Mode:{' '}
                      <span className="font-semibold text-blue-600">
                        {editMode === 'sd_inpaint'
                          ? 'SD Inpaint'
                          : 'Shape Overlay'}
                      </span>
                      {' · '}
                      Size:{' '}
                      <span className="font-semibold text-blue-600">
                        {maskSizeLevel}
                      </span>
                      {' · '}
                      Shape:{' '}
                      <span className="font-semibold text-blue-600">
                        {editMode === 'sd_inpaint'
                          ? inpaintShape
                          : overlayShape}
                      </span>
                    </p>
                  </div>
                )}

                <div className="space-y-3">
                  <button
                    onClick={handleGenerate}
                    disabled={!originalFile || isGenerating || isPredicting}
                    className="w-full inline-flex items-center justify-center rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition"
                  >
                    {isGenerating ? 'Generating...' : 'Generate Edited Image'}
                  </button>

                  {activeStep === 2 && error && (
                    <InlineAnnouncement
                      type="error"
                      message={error}
                      onClose={() => setError(null)}
                    />
                  )}

                  {activeStep === 2 &&
                    successMessage ===
                    'Edited image generated successfully. You can now run prediction.' &&
                    !error && (
                      <InlineAnnouncement
                        type="success"
                        message={successMessage}
                        onClose={() => setSuccessMessage(null)}
                      />
                    )}
                </div>
              </div>

              {/* Generated Preview */}
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-700">
                      Generated Preview
                    </h3>

                    <p className="text-xs text-slate-400 mt-1">
                      Original image, generated image, ground-truth mask, and ground-truth
                      overlay are shown below.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <PreviewCard
                    title="Original Image"
                    label="Input"
                    src={originalPreviewUrl}
                  />

                  <PreviewCard
                    title="Edited Image"
                    label="Generated"
                    src={generatedImageUrl}
                  />

                  <PreviewCard
                    title="GT Mask"
                    label="Ground Truth"
                    src={gtMaskUrl}
                  />

                  <PreviewCard
                    title="GT Overlay"
                    label="Visualization"
                    src={generatedOverlayUrl}
                  />
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Step 3 */}
        {activeStep === 3 && (
          <section className="space-y-4">
            {/* <div>
              <div className="inline-flex items-center rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 mb-3">
                Step 3
              </div>

              <h2 className="text-xl font-bold text-slate-800">
                Run Manipulation Detection
              </h2>

              <p className="text-sm text-slate-400 mt-1">
                Use the generated image from Step 2 or upload another image directly for
                prediction.
              </p>
            </div> */}

            <div className="space-y-6">
              <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-5">
                <div>
                  <h3 className="text-sm font-semibold text-slate-700 mb-3">
                    Prediction Input Source
                  </h3>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <button
                      type="button"
                      onClick={() => setPredictionSource('generated')}
                      className={[
                        'rounded-xl border px-4 py-4 text-left transition',
                        predictionSource === 'generated'
                          ? 'border-blue-300 bg-blue-50'
                          : 'border-slate-200 bg-white hover:bg-slate-50',
                      ].join(' ')}
                    >
                      <p className="text-sm font-bold text-slate-800">
                        Use Generated Image
                      </p>

                      <p className="text-xs text-slate-400 mt-1">
                        Use the edited image created in Step 2.
                      </p>
                    </button>

                    <button
                      type="button"
                      onClick={() => setPredictionSource('upload')}
                      className={[
                        'rounded-xl border px-4 py-4 text-left transition',
                        predictionSource === 'upload'
                          ? 'border-blue-300 bg-blue-50'
                          : 'border-slate-200 bg-white hover:bg-slate-50',
                      ].join(' ')}
                    >
                      <p className="text-sm font-bold text-slate-800">
                        Upload Suspected Image
                      </p>

                      <p className="text-xs text-slate-400 mt-1">
                        Upload any image and run the detection model directly.
                      </p>
                    </button>
                  </div>
                </div>

                {predictionSource === 'upload' && (
                  <div
                    onDragOver={handlePredictionDragOver}
                    onDragLeave={handlePredictionDragLeave}
                    onDrop={handlePredictionImageDrop}
                    className={[
                      'rounded-2xl border border-dashed p-5 transition',
                      isPredictionDragOver
                        ? 'border-blue-400 bg-blue-50'
                        : 'border-slate-200 bg-slate-50',
                    ].join(' ')}
                  >
                    {predictionInputPreviewUrl ? (
                      <div className="space-y-4">
                        <img
                          src={predictionInputPreviewUrl}
                          alt="Uploaded prediction input"
                          className="w-full max-h-[360px] object-contain rounded-xl border border-slate-100 bg-white"
                        />

                        <div
                          className={[
                            'rounded-xl border border-dashed px-4 py-3 text-center transition',
                            isPredictionDragOver
                              ? 'border-blue-400 bg-blue-50'
                              : 'border-slate-200 bg-white',
                          ].join(' ')}
                        >
                          <p className="text-sm font-semibold text-slate-700">
                            Drag another image here to replace current input
                          </p>
                          <p className="text-xs text-slate-400 mt-1">
                            Or use the button below to choose another file.
                          </p>
                        </div>

                        <div className="flex flex-col sm:flex-row gap-3">
                          <label className="inline-flex flex-1 items-center justify-center rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-700 cursor-pointer transition">
                            Change Image
                            <input
                              type="file"
                              accept="image/png,image/jpeg,image/jpg,image/webp"
                              className="hidden"
                              onChange={(e) => {
                                const file = e.target.files?.[0];
                                if (file) {
                                  handlePredictionImageSelect(file);
                                }
                                e.target.value = '';
                              }}
                            />
                          </label>

                          <button
                            type="button"
                            onClick={handleClearPredictionUpload}
                            className="inline-flex flex-1 items-center justify-center rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-600 hover:bg-slate-50 transition"
                          >
                            Clear Upload
                          </button>
                        </div>
                      </div>
                    ) : (
                      <label
                        className={[
                          'flex h-[260px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed transition',
                          isPredictionDragOver
                            ? 'border-blue-400 bg-blue-50'
                            : 'border-slate-300 bg-white hover:bg-blue-50',
                        ].join(' ')}
                      >
                        <p className="text-sm font-semibold text-slate-700">
                          Drag & drop image here
                        </p>

                        <p className="text-xs text-slate-400 mt-1">
                          Or click to upload PNG, JPG, JPEG, or WEBP
                        </p>

                        <input
                          type="file"
                          accept="image/png,image/jpeg,image/jpg,image/webp"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) {
                              handlePredictionImageSelect(file);
                            }
                            e.target.value = '';
                          }}
                        />
                      </label>
                    )}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
                  <div className="flex items-center justify-between gap-3 mb-3">
                    <h3 className="text-sm font-semibold text-slate-700">
                      Prediction Input Image
                    </h3>

                    <span className="text-xs font-semibold text-blue-600 bg-blue-50 border border-blue-100 rounded-full px-2 py-1">
                      {predictionSource === 'generated' ? 'Generated' : 'Uploaded'}
                    </span>
                  </div>

                  {predictionInputUrl ? (
                    <img
                      src={predictionInputUrl}
                      alt="Prediction input"
                      className="w-full max-h-[420px] object-contain rounded-xl border border-slate-100 bg-slate-50"
                    />
                  ) : (
                    <div className="h-[300px] rounded-xl border border-dashed border-slate-200 bg-slate-50 flex items-center justify-center">
                      <p className="text-sm text-slate-400 text-center">
                        Generate an edited image first or upload a custom image.
                      </p>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  {/* <div className="bg-gradient-to-br from-blue-50 to-sky-50 rounded-2xl border border-blue-100 p-5">
                    <h2 className="text-lg font-bold text-slate-800 mb-1">
                      Detection Configuration
                    </h2>

                    <p className="text-sm text-slate-500">
                      Adjust prediction parameters and run inference.
                    </p>
                  </div> */}

                  <div className="space-y-3">
                    <ControlPanel
                      threshold={threshold}
                      onThresholdChange={setThreshold}
                      usePostprocess={usePostprocess}
                      onPostprocessChange={setUsePostprocess}
                      selectedModel={selectedModel}
                      onModelChange={setSelectedModel}
                      onRunPrediction={handleRunPrediction}
                      isLoading={isPredicting}
                      disabled={!predictionInputUrl || isGenerating}
                    />

                    {activeStep === 3 && error && (
                      <InlineAnnouncement
                        type="error"
                        message={error}
                        onClose={() => setError(null)}
                      />
                    )}


                    {activeStep === 3 && successMessage && !error && (
                      <InlineAnnouncement
                        type="success"
                        message={successMessage}
                        onClose={() => setSuccessMessage(null)}
                      />
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Step 4 */}
        {activeStep === 4 && (
          <section className="space-y-8">
            <div className="space-y-4">
              <div className="flex items-center justify-between gap-4">
                <div>

                  <h2 className="text-xl font-bold text-slate-800">
                    Review Prediction Result
                  </h2>

                  <p className="text-sm text-slate-400 mt-0.5">
                    Compare the edited image, probability map, binary mask, and
                    predicted overlay.
                  </p>
                </div>

                {result && (
                  <span className="text-xs font-medium text-green-700 bg-green-50 border border-green-200 px-3 py-1 rounded-full">
                    Analysis Complete
                  </span>
                )}
              </div>

              <ResultGrid
                originalUrl={predictionInputUrl}
                probabilityMap={result?.probability_map ?? null}
                mask={result?.mask ?? null}
                overlay={result?.overlay ?? null}
              />
            </div>

            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
                <div>
                  <h2 className="text-xl font-bold text-slate-800">
                    Ground Truth Review
                  </h2>

                  <p className="text-sm text-slate-400 mt-0.5">
                    Use the generated GT mask or upload another GT mask to compare
                    with the predicted mask.
                  </p>
                </div>

                <label className="inline-flex items-center justify-center rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 cursor-pointer transition">
                  Upload GT Mask

                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/jpg,image/webp"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];

                      if (file) {
                        handleManualGtMaskSelect(file);
                      }

                      e.target.value = '';
                    }}
                  />
                </label>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
                <PreviewCard
                  title="Input Image"
                  label={predictionSource === 'generated' ? 'Generated' : 'Uploaded'}
                  src={predictionInputUrl}
                />

                <PreviewCard
                  title="GT Mask"
                  label="Ground Truth"
                  src={activeGtMaskUrl}
                />

                <PreviewCard
                  title="Predicted Mask"
                  label="Model Output"
                  src={result?.mask ?? null}
                />

                <PreviewCard
                  title="Predicted Overlay"
                  label="Visualization"
                  src={result?.overlay ?? null}
                />
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <h2 className="text-xl font-bold text-slate-800">
                  Detection Metrics
                </h2>

                <p className="text-sm text-slate-400 mt-0.5">
                  Dice, IoU, and mask area metrics calculated from GT mask and predicted mask.
                </p>
              </div>

              <MetricsSection metrics={demoMetrics} />

              {comparisonResult && (
                <div className="space-y-4 mt-8">
                  <div>
                    <h2 className="text-xl font-bold text-slate-800">
                      Model Comparison Results
                    </h2>

                    <p className="text-sm text-slate-400 mt-0.5">
                      Compare HCUNet++ and U-Net prediction masks, overlays, Dice, and IoU.
                    </p>
                  </div>

                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                    <ModelComparisonCard
                      modelName="HCUNet++"
                      modelLabel="Proposed Model"
                      inputImageSrc={predictionInputUrl}
                      result={comparisonResult.hcunetpp}
                      metrics={comparisonMetrics.hcunetpp}
                    />

                    <ModelComparisonCard
                      modelName="U-Net++"
                      modelLabel="Baseline Model"
                      inputImageSrc={predictionInputUrl}
                      result={comparisonResult.unetpp}
                      metrics={comparisonMetrics.unetpp}
                    />
                  </div>
                </div>
              )}
            </div>

            <ModelExplanation />
          </section>
        )}
      </main>

      <footer className="mt-16 border-t border-slate-100 bg-white/60 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-blue-500" />

            <span className="text-sm font-medium text-slate-600">
              Computer Vision Research Demo
            </span>
          </div>

          <div className="flex items-center gap-6 flex-wrap justify-center">
            <span className="text-xs text-slate-400">
              U-Net · ResNet34 · CBAM
            </span>

            <span className="text-xs text-slate-300">|</span>

            <span className="text-xs text-slate-400">
              Predict API:{' '}
              <code className="text-blue-500 font-mono">{API_URL}</code>
            </span>

            <span className="text-xs text-slate-300">|</span>

            <span className="text-xs text-slate-400">
              Generate API:{' '}
              <code className="text-blue-500 font-mono">
                {GENERATE_API_URL}
              </code>
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

function StepButton({
  step,
  title,
  description,
  activeStep,
  completed,
  onClick,
}: {
  step: 1 | 2 | 3 | 4;
  title: string;
  description: string;
  activeStep: 1 | 2 | 3 | 4;
  completed?: boolean;
  onClick: () => void;
}) {
  const isActive = activeStep === step;

  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'group flex items-center gap-3 rounded-xl border p-4 text-left transition',
        isActive
          ? 'border-blue-200 bg-blue-50 shadow-sm'
          : 'border-slate-100 bg-slate-50 hover:bg-white hover:border-blue-100',
      ].join(' ')}
    >
      <div
        className={[
          'flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-sm font-bold transition',
          isActive
            ? 'bg-blue-600 text-white'
            : completed
              ? 'bg-green-100 text-green-700'
              : 'bg-white text-slate-500 border border-slate-200',
        ].join(' ')}
      >
        {completed && !isActive ? '✓' : step}
      </div>

      <div className="min-w-0">
        <p
          className={[
            'text-sm font-bold transition',
            isActive ? 'text-blue-700' : 'text-slate-700',
          ].join(' ')}
        >
          Step {step}. {title}
        </p>

        <p className="text-xs text-slate-400 mt-0.5 truncate">
          {description}
        </p>
      </div>
    </button>
  );
}

function formatMetricValue(value?: number | null) {
  if (value === null || value === undefined) return '-';
  return value.toFixed(3);
}

function formatAreaValue(value?: number | null) {
  if (value === null || value === undefined) return '-';
  return `${(value * 100).toFixed(2)}%`;
}

function ModelComparisonCard({
  modelName,
  modelLabel,
  inputImageSrc,
  result,
  metrics,
}: {
  modelName: string;
  modelLabel: string;
  inputImageSrc: string | null;
  result: PredictionResponse | null;
  metrics: DemoMetrics | null;
}) {
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-bold text-slate-800">{modelName}</h3>
          <p className="text-sm text-slate-400">{modelLabel}</p>
        </div>

        <span className="text-xs font-semibold text-blue-600 bg-blue-50 border border-blue-100 rounded-full px-3 py-1">
          Model Output
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <PreviewCard
          title="Prediction Input Image"
          label="Input"
          src={inputImageSrc}
        />

        <PreviewCard
          title="Predicted Overlay"
          label="Overlay"
          src={result?.overlay ?? null}
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="rounded-xl bg-slate-50 border border-slate-100 px-4 py-3">
          <p className="text-xs font-semibold text-slate-400">Dice</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">
            {formatMetricValue(metrics?.dice)}
          </p>
        </div>

        <div className="rounded-xl bg-slate-50 border border-slate-100 px-4 py-3">
          <p className="text-xs font-semibold text-slate-400">IoU</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">
            {formatMetricValue(metrics?.iou)}
          </p>
        </div>

        <div className="rounded-xl bg-slate-50 border border-slate-100 px-4 py-3">
          <p className="text-xs font-semibold text-slate-400">Pred Area</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">
            {formatAreaValue(metrics?.pred_area)}
          </p>
        </div>
      </div>
    </div>
  );
}

function PreviewCard({
  title,
  label,
  src,
}: {
  title: string;
  label: string;
  src: string | null;
}) {
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-4">
      <div className="flex items-start justify-between gap-2 mb-3">
        <h3 className="text-sm font-bold text-slate-700 leading-snug">
          {title}
        </h3>

        <span className="text-xs font-semibold text-blue-600 bg-blue-50 border border-blue-100 rounded-full px-2 py-1 whitespace-nowrap">
          {label}
        </span>
      </div>

      {src ? (
        <img
          src={src}
          alt={title}
          className="w-full h-[260px] object-contain rounded-xl border border-slate-100 bg-slate-50"
        />
      ) : (
        <div className="w-full h-[220px] rounded-xl border border-dashed border-slate-200 bg-slate-50 flex items-center justify-center">
          <p className="text-sm text-slate-400 text-center">Awaiting image</p>
        </div>
      )}
    </div>
  );
}

function InlineAnnouncement({
  type,
  message,
  onClose,
}: {
  type: 'success' | 'error';
  message: string;
  onClose: () => void;
}) {
  const isError = type === 'error';

  return (
    <div
      className={[
        'flex items-start gap-3 rounded-xl border px-4 py-3',
        isError
          ? 'bg-red-50 border-red-200 text-red-700'
          : 'bg-green-50 border-green-200 text-green-700',
      ].join(' ')}
    >
      {isError ? (
        <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
      ) : (
        <CheckCircle2 className="w-5 h-5 shrink-0 mt-0.5" />
      )}

      <p className="text-sm font-medium flex-1 whitespace-pre-wrap">
        {message}
      </p>

      <button
        type="button"
        onClick={onClose}
        className={[
          'transition-colors',
          isError
            ? 'text-red-400 hover:text-red-600'
            : 'text-green-400 hover:text-green-600',
        ].join(' ')}
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}