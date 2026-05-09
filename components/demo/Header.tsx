import { Cpu, Layers } from 'lucide-react';

export default function Header() {
  return (
    <header className="relative overflow-hidden bg-gradient-to-br from-blue-700 via-blue-600 to-sky-500 text-white">
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-0 left-0 w-96 h-96 bg-white rounded-full -translate-x-48 -translate-y-48" />
        <div className="absolute bottom-0 right-0 w-80 h-80 bg-sky-200 rounded-full translate-x-32 translate-y-32" />
      </div>
      <div className="relative max-w-7xl mx-auto px-6 py-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-white/20 rounded-xl backdrop-blur-sm">
            <Cpu className="w-7 h-7 text-white" />
          </div>
          <span className="text-sm font-medium text-blue-100 uppercase tracking-wider">
            Computer Vision Research
          </span>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-3">
          Image Manipulation
          <span className="block text-sky-200">Localization Demo</span>
        </h1>
        <p className="text-blue-100 text-lg mb-6 max-w-2xl leading-relaxed">
          Visualizing AI-based manipulated region detection with deep learning segmentation models.
        </p>
        <div className="inline-flex items-center gap-2 bg-white/15 backdrop-blur-sm border border-white/25 rounded-full px-4 py-2">
          <Layers className="w-4 h-4 text-sky-200" />
          <span className="text-sm font-medium text-white">
            U-Net++ ResNet34 + High-Frequency Input + CBAM
          </span>
        </div>
      </div>
    </header>
  );
}
