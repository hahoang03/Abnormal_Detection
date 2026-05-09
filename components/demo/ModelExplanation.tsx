import { ChevronRight, Image as ImageIconLucide, Cpu, Waves, Network, ChartBar as BarChart2, Filter, Layers } from 'lucide-react';

const steps = [
  {
    icon: <ImageIconLucide className="w-5 h-5" />,
    label: 'Input Image',
    description: 'Accepts any JPEG/PNG',
    color: 'bg-blue-50 text-blue-600 border-blue-100',
    dot: 'bg-blue-500',
  },
  {
    icon: <Cpu className="w-5 h-5" />,
    label: 'Resize & Normalize',
    description: 'Standardized preprocessing',
    color: 'bg-sky-50 text-sky-600 border-sky-100',
    dot: 'bg-sky-500',
  },
  {
    icon: <Waves className="w-5 h-5" />,
    label: 'High-Frequency Extraction',
    description: 'Edge & noise artifacts',
    color: 'bg-cyan-50 text-cyan-600 border-cyan-100',
    dot: 'bg-cyan-500',
  },
  {
    icon: <Network className="w-5 h-5" />,
    label: 'U-Net++ Segmentation',
    description: 'ResNet34 encoder + CBAM',
    color: 'bg-blue-50 text-blue-700 border-blue-100',
    dot: 'bg-blue-700',
  },
  {
    icon: <BarChart2 className="w-5 h-5" />,
    label: 'Probability Map',
    description: 'Per-pixel manipulation score',
    color: 'bg-indigo-50 text-indigo-600 border-indigo-100',
    dot: 'bg-indigo-500',
  },
  {
    icon: <Filter className="w-5 h-5" />,
    label: 'Thresholding',
    description: 'Binarize with threshold τ',
    color: 'bg-sky-50 text-sky-700 border-sky-100',
    dot: 'bg-sky-700',
  },
  {
    icon: <Layers className="w-5 h-5" />,
    label: 'Final Overlay',
    description: 'Highlighted manipulated regions',
    color: 'bg-blue-50 text-blue-600 border-blue-100',
    dot: 'bg-blue-600',
  },
];

export default function ModelExplanation() {
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-slate-800 mb-1">How the Model Works</h2>
        <p className="text-slate-500 text-sm">
          End-to-end pipeline from raw image to manipulated region localization.
        </p>
      </div>

      {/* Desktop flow */}
      <div className="hidden md:flex items-start gap-0 overflow-x-auto pb-2">
        {steps.map((step, i) => (
          <div key={step.label} className="flex items-start">
            <div className="flex flex-col items-center min-w-[120px] max-w-[120px]">
              <div className={`w-11 h-11 rounded-xl border flex items-center justify-center mb-2 shrink-0 ${step.color}`}>
                {step.icon}
              </div>
              <p className="text-xs font-semibold text-slate-700 text-center leading-tight mb-1">{step.label}</p>
              <p className="text-[10px] text-slate-400 text-center leading-tight">{step.description}</p>
            </div>
            {i < steps.length - 1 && (
              <div className="flex items-center mt-3 mx-1 shrink-0">
                <ChevronRight className="w-5 h-5 text-slate-300" />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Mobile flow */}
      <div className="md:hidden space-y-3">
        {steps.map((step, i) => (
          <div key={step.label} className="flex items-start gap-3">
            <div className="flex flex-col items-center">
              <div className={`w-9 h-9 rounded-xl border flex items-center justify-center shrink-0 ${step.color}`}>
                {step.icon}
              </div>
              {i < steps.length - 1 && (
                <div className={`w-0.5 h-6 mt-1 rounded-full ${step.dot} opacity-30`} />
              )}
            </div>
            <div className="pt-1.5">
              <p className="text-sm font-semibold text-slate-700">{step.label}</p>
              <p className="text-xs text-slate-400">{step.description}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Architecture note */}
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-3">
        {[
          {
            label: 'Backbone',
            value: 'ResNet34',
            desc: 'Pre-trained ImageNet encoder',
          },
          {
            label: 'Architecture',
            value: 'U-Net++',
            desc: 'Nested dense skip connections',
          },
          {
            label: 'Attention',
            value: 'CBAM',
            desc: 'Channel + spatial attention modules',
          },
        ].map((item) => (
          <div key={item.label} className="bg-slate-50 rounded-xl p-4 border border-slate-100">
            <p className="text-xs text-slate-400 uppercase tracking-wide font-medium mb-1">{item.label}</p>
            <p className="text-base font-bold text-blue-700">{item.value}</p>
            <p className="text-xs text-slate-500 mt-0.5">{item.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
