'use client';

import { Image as ImageIcon } from 'lucide-react';
import { ResultPanelProps } from '@/types/demo';

function ResultPanel({ title, src, placeholder, badge }: ResultPanelProps) {
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden group hover:shadow-md hover:border-blue-100 transition-all duration-200">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-50 bg-slate-50/70">
        <h4 className="text-sm font-semibold text-slate-700">{title}</h4>
        {badge && (
          <span className="text-xs text-blue-600 bg-blue-50 border border-blue-100 px-2 py-0.5 rounded-full font-medium">
            {badge}
          </span>
        )}
      </div>
      <div className="aspect-square relative bg-slate-50/50 flex items-center justify-center">
        {src ? (
          <img
            src={src}
            alt={title}
            className="w-full h-full object-contain transition-opacity duration-300"
          />
        ) : (
          <div className="flex flex-col items-center gap-3 text-slate-300 p-8">
            <div className="p-4 bg-slate-100 rounded-2xl">
              <ImageIcon className="w-8 h-8" />
            </div>
            <p className="text-xs text-center text-slate-400 max-w-[120px]">
              {placeholder || 'Run prediction to see results'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

interface ResultGridProps {
  originalUrl: string | null;
  probabilityMap: string | null;
  mask: string | null;
  overlay: string | null;
}

export default function ResultGrid({ originalUrl, probabilityMap, mask, overlay }: ResultGridProps) {
  const panels: ResultPanelProps[] = [
    {
      title: 'Original Image',
      src: originalUrl,
      placeholder: 'Upload an image',
      badge: 'Input',
    },
    {
      title: 'Probability Map',
      src: probabilityMap ? `data:image/png;base64,${probabilityMap}` : null,
      placeholder: 'Awaiting prediction',
      badge: 'Raw Output',
    },
    {
      title: 'Predicted Mask',
      src: mask ? `data:image/png;base64,${mask}` : null,
      placeholder: 'Awaiting prediction',
      badge: 'Threshold Applied',
    },
    {
      title: 'Overlay',
      src: overlay ? `data:image/png;base64,${overlay}` : null,
      placeholder: 'Awaiting prediction',
      badge: 'Visualization',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
      {panels.map((panel) => (
        <ResultPanel key={panel.title} {...panel} />
      ))}
    </div>
  );
}
