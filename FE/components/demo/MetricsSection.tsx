'use client';

type Metrics = {
  dice: number;
  iou: number;
  gt_area: number;
  pred_area: number;
  gt_pixels: number;
  pred_pixels: number;
  threshold: number;
} | null;

interface MetricsSectionProps {
  metrics: Metrics;
}

function formatNumber(value: number) {
  return value.toFixed(3);
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function MetricCard({
  title,
  value,
  description,
}: {
  title: string;
  value: string;
  description: string;
}) {
  return (
    <div className="w-full h-full bg-white rounded-2xl border border-slate-100 shadow-sm p-5">
      <p className="text-sm font-semibold text-slate-500">{title}</p>

      <p className="text-3xl font-bold text-slate-800 mt-2">{value}</p>

      <p className="text-xs text-slate-400 mt-2 leading-relaxed">
        {description}
      </p>
    </div>
  );
}

export default function MetricsSection({ metrics }: MetricsSectionProps) {
  if (!metrics) {
    return (
      <div className="bg-white rounded-2xl border border-dashed border-slate-200 p-8 text-center">
        <p className="text-sm text-slate-400">
          Run prediction to calculate Dice, IoU, GT area, and predicted area.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full">
      <MetricCard
        title="Dice Score"
        value={formatNumber(metrics.dice)}
        description="Overlap score between predicted mask and GT mask. Higher is better."
      />

      <MetricCard
        title="IoU"
        value={formatNumber(metrics.iou)}
        description="Intersection over Union between predicted region and GT region. Higher is better."
      />

      {/* <MetricCard
        title="GT Area"
        value={formatPercent(metrics.gt_area)}
        description={`Ground-truth edited region size. Pixels: ${metrics.gt_pixels}`}
      />

      <MetricCard
        title="Pred Area"
        value={formatPercent(metrics.pred_area)}
        description={`Predicted edited region size. Pixels: ${metrics.pred_pixels}. Threshold: ${metrics.threshold}`}
      /> */}
    </div>
  );
}