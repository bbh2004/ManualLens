import React from 'react';

interface StatusBadgeProps {
  status: 'processing' | 'ready' | 'error' | string;
  stage?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, stage }) => {
  const getStageLabel = (s?: string) => {
    switch (s) {
      case 'extracting_text':
        return 'Extracting Text';
      case 'indexing_text':
        return 'Indexing Text & BM25';
      case 'rendering_pages':
        return 'Rendering Pages';
      case 'indexing_visuals':
        return 'Indexing Visuals';
      case 'ready':
        return 'Ready';
      default:
        return s || status;
    }
  };

  if (status === 'ready') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 border border-emerald-200">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
        Ready
      </span>
    );
  }

  if (status === 'processing') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 border border-blue-200">
        <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse"></span>
        {stage ? getStageLabel(stage) : 'Processing...'}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-rose-100 text-rose-800 border border-rose-200">
      <span className="w-1.5 h-1.5 rounded-full bg-rose-500"></span>
      Failed
    </span>
  );
};

