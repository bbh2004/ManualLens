import React from 'react';
import { CheckCircle, AlertTriangle, FileText, Image as ImageIcon, ExternalLink } from 'lucide-react';
import type { QueryResponse } from '../api/client';

interface AnswerCardProps {
  question: string;
  response: QueryResponse;
  documentId: string;
  onOpenPage: (pageNumber: number) => void;
}

export const AnswerCard: React.FC<AnswerCardProps> = ({
  question,
  response,
  onOpenPage,
}) => {
  const { answer, citations, relevant_pages, uncertainties } = response;

  const renderFormattedAnswer = (text: string) => {
    // Highlight [Page X] citations and make them clickable
    const parts = text.split(/(\[Page\s+\d+\])/gi);

    return parts.map((part, index) => {
      const match = part.match(/\[Page\s+(\d+)\]/i);
      if (match) {
        const pageNum = parseInt(match[1], 10);
        return (
          <button
            key={index}
            onClick={() => onOpenPage(pageNum)}
            className="inline-flex items-center gap-0.5 mx-1 px-1.5 py-0.5 text-xs font-semibold rounded bg-indigo-100 text-indigo-700 hover:bg-indigo-200 transition-colors border border-indigo-200"
            title={`View Page ${pageNum}`}
          >
            {part}
            <ExternalLink className="w-2.5 h-2.5 inline" />
          </button>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 space-y-5 animate-fade-in">
      {/* Question Header */}
      <div className="border-b border-slate-100 pb-3">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Question</span>
        <h3 className="text-base font-semibold text-slate-900 mt-0.5">{question}</h3>
      </div>

      {/* Answer Body */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <CheckCircle className="w-4 h-4 text-emerald-600" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Grounded Answer
          </span>
        </div>
        <div className="text-slate-800 text-sm leading-relaxed whitespace-pre-wrap">
          {renderFormattedAnswer(answer)}
        </div>
      </div>

      {/* Citations List */}
      {citations && citations.length > 0 && (
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200/80">
          <div className="flex items-center gap-2 mb-3">
            <FileText className="w-4 h-4 text-indigo-600" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-600">
              Retrieved Text Citations ({citations.length})
            </span>
          </div>

          <div className="grid grid-cols-1 gap-2.5">
            {citations.map((c, i) => (
              <div
                key={i}
                onClick={() => onOpenPage(c.page_number)}
                className="bg-white p-3 rounded-lg border border-slate-200 hover:border-indigo-300 transition-colors cursor-pointer group"
              >
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-semibold text-indigo-600 group-hover:underline flex items-center gap-1">
                    Page {c.page_number}
                    <ExternalLink className="w-3 h-3 text-slate-400" />
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600 uppercase">
                    {c.source}
                  </span>
                </div>
                <p className="text-xs text-slate-600 italic line-clamp-3">
                  "{c.chunk_text}"
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Visual Pages (Phase 2 preview / thumbnails) */}
      {relevant_pages && relevant_pages.length > 0 && (
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200/80">
          <div className="flex items-center gap-2 mb-3">
            <ImageIcon className="w-4 h-4 text-indigo-600" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-600">
              Relevant Visual Pages & Diagrams ({relevant_pages.length})
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {relevant_pages.map((vp, idx) => (
              <div
                key={idx}
                onClick={() => onOpenPage(vp.page_number)}
                className="group relative bg-white border border-slate-200 rounded-lg overflow-hidden shadow-sm hover:border-indigo-500 cursor-pointer transition-all"
              >
                <div className="aspect-[3/4] bg-slate-100 flex items-center justify-center overflow-hidden">
                  <img
                    src={vp.image_url}
                    alt={`Page ${vp.page_number}`}
                    className="w-full h-full object-cover object-top group-hover:scale-105 transition-transform duration-200"
                    loading="lazy"
                  />
                </div>
                <div className="p-2 bg-white text-xs flex items-center justify-between border-t border-slate-100">
                  <span className="font-medium text-slate-800">Page {vp.page_number}</span>
                  {vp.visual_score > 0 && (
                    <span className="text-[10px] text-slate-400">
                      Score: {vp.visual_score.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Uncertainty Flags */}
      {uncertainties && uncertainties.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3.5 text-xs text-amber-800 flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-semibold text-amber-900">Uncertainties & Manual Coverage Gaps:</div>
            <ul className="list-disc list-inside space-y-0.5 text-amber-800">
              {uncertainties.map((u, i) => (
                <li key={i}>{u}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};
