'use client';

import { useState, useCallback, useEffect } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import Link from 'next/link';
import {
    Upload,
    Download,
    Wand2,
    CircleAlert as AlertCircle,
    CircleCheck as CheckCircle2,
    X,
} from 'lucide-react';
import Header from '@/components/demo/Header';

const GENERATE_API_URL =
    process.env.NEXT_PUBLIC_GENERATE_API_URL || 'http://localhost:8000';

type GenerateMode = 'clean' | 'sd_inpaint' | 'shape_overlay';

type GenerateResponse = {
    original: string;
    edited: string;
    mask: string;
    overlay: string;
    info: {
        edit_group: string;
        edit_type: string;
        level: string;
        shape: string;
        area_ratio: number;
        prompt: string;
        cue_mode: string;
        strength: number;
        guidance_scale: number;
        seed: number;
        retry_count: number;
        status: string;
        diff_mean: number;
        diff_median: number;
        diff_p75: number;
        diff_p90: number;
        changed_ratio_5: number;
        changed_ratio_10: number;
        changed_ratio_20: number;
        opacity?: number;
        edge_blur?: number;
        overlay_color?: number[];
    };
};

export default function GeneratePage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [mode, setMode] = useState<GenerateMode>('sd_inpaint');
    const [useSubtleCue, setUseSubtleCue] = useState(true);
    const [result, setResult] = useState<GenerateResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isDragging, setIsDragging] = useState(false);

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

    const validateAndSelectImage = useCallback(
        (file: File) => {
            if (!file.type.startsWith('image/')) {
                setError('Please upload a valid image file.');
                return;
            }

            handleImageSelect(file);
        },
        [handleImageSelect]
    );

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];

        if (!file) return;

        validateAndSelectImage(file);

        event.target.value = '';
    };

    const handleDragOver = (
        event: DragEvent<HTMLLabelElement | HTMLDivElement>
    ) => {
        event.preventDefault();
        event.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (
        event: DragEvent<HTMLLabelElement | HTMLDivElement>
    ) => {
        event.preventDefault();
        event.stopPropagation();
        setIsDragging(false);
    };

    const handleDrop = (
        event: DragEvent<HTMLLabelElement | HTMLDivElement>
    ) => {
        event.preventDefault();
        event.stopPropagation();
        setIsDragging(false);

        const file = event.dataTransfer.files?.[0];

        if (!file) return;

        validateAndSelectImage(file);
    };

    const handleClear = useCallback(() => {
        setSelectedFile(null);

        setPreviewUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return null;
        });

        setResult(null);
        setError(null);
        setIsDragging(false);
    }, []);

    const handleGenerate = async () => {
        if (!selectedFile) return;

        setIsLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('mode', mode);
        formData.append(
            'use_subtle_cue',
            (mode === 'sd_inpaint' && useSubtleCue).toString()
        );

        try {
            const response = await fetch(`${GENERATE_API_URL}/api/generate`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            setResult(data);
        } catch (err) {
            console.error(err);
            setError('Generate failed. Please check that the backend server is running.');
        } finally {
            setIsLoading(false);
        }
    };

    const downloadImage = (dataUrl: string, filename: string) => {
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    useEffect(() => {
        return () => {
            if (previewUrl) {
                URL.revokeObjectURL(previewUrl);
            }
        };
    }, [previewUrl]);

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-sky-50/20">
            <Header />

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                        <Link
                            href="/"
                            className="text-sm font-medium text-blue-600 hover:text-blue-700"
                        >
                            ← Back to Detection Page
                        </Link>

                        <h1 className="text-2xl font-bold text-slate-800 mt-3">
                            Generate Abnormal Image
                        </h1>

                        <p className="text-sm text-slate-400 mt-1">
                            Create clean, SD inpaint, or shape overlay samples for demo and testing.
                        </p>
                    </div>

                    <div className="text-xs text-slate-400">
                        API:{' '}
                        <code className="text-blue-500 font-mono">
                            {GENERATE_API_URL}
                        </code>
                    </div>
                </div>

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

                {result && !error && (
                    <div className="flex items-center gap-3 bg-green-50 border border-green-200 text-green-700 rounded-2xl px-5 py-4">
                        <CheckCircle2 className="w-5 h-5 shrink-0" />

                        <p className="text-sm font-medium">
                            Image generation complete. You can download the edited image, mask, and overlay.
                        </p>
                    </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <section className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-4">
                        <div>
                            <h2 className="text-lg font-bold text-slate-800 mb-1">
                                Upload Image
                            </h2>

                            <p className="text-sm text-slate-400">
                                Select or drag an image to generate a manipulated version.
                            </p>
                        </div>

                        {!previewUrl ? (
                            <label
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                                className={`flex flex-col items-center justify-center border-2 border-dashed rounded-2xl p-10 cursor-pointer transition ${isDragging
                                    ? 'border-blue-500 bg-blue-50'
                                    : 'border-slate-200 hover:border-blue-300 hover:bg-blue-50/40'
                                    }`}
                            >
                                <Upload className="w-10 h-10 text-blue-500 mb-3" />

                                <span className="text-sm font-medium text-slate-700">
                                    {isDragging ? 'Drop image here' : 'Click or drag image here'}
                                </span>

                                <span className="text-xs text-slate-400 mt-1">
                                    JPG, PNG, WEBP supported
                                </span>

                                <input
                                    type="file"
                                    accept="image/*"
                                    onChange={handleFileChange}
                                    className="hidden"
                                />
                            </label>
                        ) : (
                            <div
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                                className={`space-y-4 rounded-2xl transition ${isDragging ? 'ring-2 ring-blue-300 bg-blue-50/40 p-2' : ''
                                    }`}
                            >
                                <img
                                    src={previewUrl}
                                    alt="Preview"
                                    className="w-full rounded-2xl border border-slate-100"
                                />

                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                    <label className="w-full rounded-xl border border-slate-200 text-slate-600 py-2 text-sm font-medium hover:bg-slate-50 text-center cursor-pointer">
                                        Change Image

                                        <input
                                            type="file"
                                            accept="image/*"
                                            onChange={handleFileChange}
                                            className="hidden"
                                        />
                                    </label>

                                    <button
                                        onClick={handleClear}
                                        className="w-full rounded-xl border border-slate-200 text-slate-600 py-2 text-sm font-medium hover:bg-slate-50"
                                    >
                                        Clear Image
                                    </button>
                                </div>

                                <p className="text-xs text-slate-400 text-center">
                                    You can also drag another image here to replace the current one.
                                </p>
                            </div>
                        )}
                    </section>

                    <section className="space-y-4">
                        <div className="bg-gradient-to-br from-blue-50 to-sky-50 rounded-2xl border border-blue-100 p-5">
                            <h2 className="text-lg font-bold text-slate-800 mb-1">
                                Generation Settings
                            </h2>

                            <p className="text-sm text-slate-500">
                                Choose the manipulation type based on your dataset generation logic.
                            </p>
                        </div>

                        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-5">
                            <div>
                                <label className="block text-sm font-semibold text-slate-700 mb-2">
                                    Edit Mode
                                </label>

                                <select
                                    value={mode}
                                    onChange={(event) =>
                                        setMode(event.target.value as GenerateMode)
                                    }
                                    className="w-full rounded-xl border border-slate-200 px-4 py-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-200"
                                >
                                    <option value="clean">Clean</option>
                                    <option value="sd_inpaint">SD Inpaint</option>
                                    <option value="shape_overlay">Shape Overlay</option>
                                </select>
                            </div>

                            {mode === 'sd_inpaint' && (
                                <label className="flex items-center gap-3 text-sm text-slate-600">
                                    <input
                                        type="checkbox"
                                        checked={useSubtleCue}
                                        onChange={(event) =>
                                            setUseSubtleCue(event.target.checked)
                                        }
                                        className="w-4 h-4"
                                    />
                                    Apply subtle statistical cue
                                </label>
                            )}

                            <button
                                onClick={handleGenerate}
                                disabled={!selectedFile || isLoading}
                                className="w-full flex items-center justify-center gap-2 rounded-xl bg-blue-600 text-white py-3 text-sm font-bold hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition"
                            >
                                <Wand2 className="w-4 h-4" />
                                {isLoading ? 'Generating...' : 'Generate Image'}
                            </button>
                        </div>
                    </section>
                </div>

                {result && (
                    <>
                        <section className="space-y-4">
                            <div>
                                <h2 className="text-xl font-bold text-slate-800">
                                    Generated Results
                                </h2>

                                <p className="text-sm text-slate-400 mt-0.5">
                                    Original, edited image, ground-truth mask, and overlay visualization.
                                </p>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
                                <ImageCard
                                    title="Original"
                                    src={result.original}
                                    onDownload={() =>
                                        downloadImage(result.original, 'original.png')
                                    }
                                />

                                <ImageCard
                                    title="Edited"
                                    src={result.edited}
                                    onDownload={() =>
                                        downloadImage(
                                            result.edited,
                                            `${result.info.edit_group}_edited.png`
                                        )
                                    }
                                />

                                <ImageCard
                                    title="GT Mask"
                                    src={result.mask}
                                    onDownload={() =>
                                        downloadImage(
                                            result.mask,
                                            `${result.info.edit_group}_mask.png`
                                        )
                                    }
                                />

                                <ImageCard
                                    title="Overlay"
                                    src={result.overlay}
                                    onDownload={() =>
                                        downloadImage(
                                            result.overlay,
                                            `${result.info.edit_group}_overlay.png`
                                        )
                                    }
                                />
                            </div>
                        </section>

                        <section className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
                            <h2 className="text-xl font-bold text-slate-800 mb-4">
                                Generation Info
                            </h2>

                            <pre className="text-sm text-slate-600 whitespace-pre-wrap leading-7 bg-slate-50 border border-slate-100 rounded-2xl p-5 overflow-x-auto">
                                {`Edit group: ${result.info.edit_group}
Edit type: ${result.info.edit_type}
Status: ${result.info.status}
Mask shape: ${result.info.shape}
Mask level: ${result.info.level}
Area ratio: ${result.info.area_ratio.toFixed(3)}

Prompt: ${result.info.prompt}
Cue mode: ${result.info.cue_mode}
Strength: ${result.info.strength.toFixed(3)}
Guidance scale: ${result.info.guidance_scale.toFixed(3)}
Seed: ${result.info.seed}
Retry count: ${result.info.retry_count}

Diff mean: ${result.info.diff_mean.toFixed(2)}
Diff median: ${result.info.diff_median.toFixed(2)}
Diff p75: ${result.info.diff_p75.toFixed(2)}
Diff p90: ${result.info.diff_p90.toFixed(2)}
Changed ratio >5: ${result.info.changed_ratio_5.toFixed(2)}
Changed ratio >10: ${result.info.changed_ratio_10.toFixed(2)}
Changed ratio >20: ${result.info.changed_ratio_20.toFixed(2)}

Shape opacity: ${result.info.opacity ?? 'none'}
Shape edge blur: ${result.info.edge_blur ?? 'none'}
Shape color: ${result.info.overlay_color
                                        ? result.info.overlay_color.join(', ')
                                        : 'none'
                                    }`}
                            </pre>
                        </section>
                    </>
                )}
            </main>
        </div>
    );
}

function ImageCard({
    title,
    src,
    onDownload,
}: {
    title: string;
    src: string;
    onDownload: () => void;
}) {
    return (
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-4 space-y-3">
            <h3 className="font-bold text-slate-800">{title}</h3>

            <img
                src={src}
                alt={title}
                className="w-full rounded-xl border border-slate-100"
            />

            <button
                onClick={onDownload}
                className="w-full flex items-center justify-center gap-2 rounded-xl border border-slate-200 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-50 transition"
            >
                <Download className="w-4 h-4" />
                Download
            </button>
        </div>
    );
}