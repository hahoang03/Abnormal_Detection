import { Layers } from 'lucide-react';

export default function Header() {
  return (
    <header className="relative overflow-hidden bg-gradient-to-br from-blue-700 via-blue-600 to-sky-500 text-white">
      {/* Soft background decoration */}
      <div className="pointer-events-none absolute inset-0 opacity-[0.06]">
        <div className="absolute top-0 left-0 h-56 w-56 -translate-x-32 -translate-y-32 rounded-full bg-white" />
        <div className="absolute bottom-0 right-0 h-60 w-60 translate-x-28 translate-y-28 rounded-full bg-sky-200" />
      </div>

      <div className="relative mx-auto max-w-7xl px-6 py-6 md:py-7">
        <div className="flex flex-col items-center text-center">
          <h1 className="max-w-4xl text-2xl font-bold leading-tight tracking-tight md:text-3xl lg:text-4xl">
            Abnormal Detection in Artworks
          </h1>

          {/* <p className="mt-2 max-w-3xl text-sm font-medium leading-relaxed text-blue-100 md:text-base">
            Visualizing manipulated region detection with deep learning segmentation models.
          </p> */}

          <div className="mt-4 inline-flex max-w-full items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3.5 py-1.5 backdrop-blur-sm">
            <Layers className="h-4 w-4 shrink-0 text-sky-200" />

            <span className="text-xs font-medium text-white md:text-sm">
              U-Net++ ResNet34 + High-Frequency Input + CBAM + EMA
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}