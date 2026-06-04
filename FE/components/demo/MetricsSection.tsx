import { ChartBar as BarChart2, TrendingUp, TrendingDown, Target, FileSliders as Sliders } from 'lucide-react';
import { MetricCardProps } from '@/types/demo';

function MetricCard({ label, value, decimals = 4, unit, description }: MetricCardProps) {
  const displayValue =
    typeof value === 'number'
      ? decimals === 0
        ? value.toString()
        : value.toFixed(decimals)
      : value;

  const icons: Record<string, React.ReactNode> = {
    'Raw Prob Min': <TrendingDown className="w-4 h-4 text-sky-500" />,
    'Raw Prob Mean': <BarChart2 className="w-4 h-4 text-blue-500" />,
    'Raw Prob Max': <TrendingUp className="w-4 h-4 text-blue-600" />,
    'Pred Area Ratio': <Target className="w-4 h-4 text-cyan-500" />,
    'Threshold': <Sliders className="w-4 h-4 text-slate-500" />,
  };

  const bgColors: Record<string, string> = {
    'Raw Prob Min': 'bg-sky-50 border-sky-100',
    'Raw Prob Mean': 'bg-blue-50 border-blue-100',
    'Raw Prob Max': 'bg-blue-50 border-blue-100',
    'Pred Area Ratio': 'bg-cyan-50 border-cyan-100',
    'Threshold': 'bg-slate-50 border-slate-100',
  };

  return (
    <div className={`rounded-2xl border p-4 ${bgColors[label] || 'bg-white border-slate-100'} transition-all hover:shadow-sm`}>
      <div className="flex items-center gap-2 mb-2">
        <div className="p-1.5 bg-white rounded-lg shadow-sm">{icons[label] || <BarChart2 className="w-4 h-4 text-blue-500" />}</div>
        <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">{label}</span>
      </div>
      <div className="flex items-end gap-1">
        <span className="text-2xl font-bold text-slate-800 tabular-nums">{displayValue}</span>
        {unit && <span className="text-sm text-slate-400 mb-0.5">{unit}</span>}
      </div>
      {description && (
        <p className="text-xs text-slate-400 mt-1">{description}</p>
      )}
    </div>
  );
}

interface MetricsSectionProps {
  metrics: {
    raw_prob_min: number;
    raw_prob_mean: number;
    raw_prob_max: number;
    pred_area_ratio: number;
    threshold: number;
  } | null;
}

export default function MetricsSection({ metrics }: MetricsSectionProps) {
  if (!metrics) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {['Raw Prob Min', 'Raw Prob Mean', 'Raw Prob Max', 'Pred Area Ratio', 'Threshold'].map((label) => (
          <div key={label} className="rounded-2xl border border-slate-100 bg-slate-50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-7 h-7 bg-slate-100 rounded-lg" />
              <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">{label}</span>
            </div>
            <div className="h-8 bg-slate-100 rounded-lg w-24 animate-pulse" />
          </div>
        ))}
      </div>
    );
  }

  const cards: MetricCardProps[] = [
    {
      label: 'Raw Prob Min',
      value: metrics.raw_prob_min,
      decimals: 4,
      description: 'Minimum predicted probability',
    },
    {
      label: 'Raw Prob Mean',
      value: metrics.raw_prob_mean,
      decimals: 4,
      description: 'Mean predicted probability',
    },
    {
      label: 'Raw Prob Max',
      value: metrics.raw_prob_max,
      decimals: 4,
      description: 'Maximum predicted probability',
    },
    {
      label: 'Pred Area Ratio',
      value: metrics.pred_area_ratio,
      decimals: 4,
      description: 'Fraction of image flagged',
    },
    {
      label: 'Threshold',
      value: metrics.threshold,
      decimals: 2,
      description: 'Applied decision boundary',
    },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
      {cards.map((card) => (
        <MetricCard key={card.label} {...card} />
      ))}
    </div>
  );
}
