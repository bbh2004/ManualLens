import React from 'react';
import { X, ZoomIn, Download, ExternalLink } from 'lucide-react';
import { getPageUrl } from '../api/client';

interface PageViewerProps {
  documentId: string;
  pageNumber: number | null;
  onClose: () => void;
}

export const PageViewer: React.FC<PageViewerProps> = ({ documentId, pageNumber, onClose }) => {
  if (pageNumber === null) return null;

  const imageUrl = getPageUrl(documentId, pageNumber);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 animate-fade-in">
      <div className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[92vh] flex flex-col overflow-hidden border border-slate-700">
        {/* Modal Header */}
        <div className="px-5 py-3 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2">
            <ZoomIn className="w-5 h-5 text-indigo-600" />
            <span className="font-semibold text-slate-800 text-sm">
              Page Inspection — Page {pageNumber}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <a
              href={imageUrl}
              target="_blank"
              rel="noreferrer"
              className="p-1.5 text-slate-500 hover:text-slate-800 hover:bg-slate-200 rounded-md transition-colors"
              title="Open full image in new tab"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
            <a
              href={imageUrl}
              download={`doc_${documentId}_page_${pageNumber}.png`}
              className="p-1.5 text-slate-500 hover:text-slate-800 hover:bg-slate-200 rounded-md transition-colors"
              title="Download image"
            >
              <Download className="w-4 h-4" />
            </a>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-500 hover:text-slate-800 hover:bg-slate-200 rounded-md transition-colors"
              title="Close viewer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Image Display */}
        <div className="flex-1 overflow-auto p-4 bg-slate-900/95 flex items-center justify-center">
          <img
            src={imageUrl}
            alt={`Page ${pageNumber}`}
            className="max-h-[80vh] w-auto object-contain rounded shadow-lg border border-slate-800 bg-white"
            onError={(e) => {
              (e.target as HTMLElement).style.display = 'none';
            }}
          />
        </div>
      </div>
    </div>
  );
};

