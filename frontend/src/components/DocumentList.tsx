import React, { useState } from 'react';
import { BookOpen, Trash2, Clock, Layers, Loader2 } from 'lucide-react';
import type { DocumentInfo } from '../api/client';
import { deleteDocument } from '../api/client';
import { StatusBadge } from './StatusBadge';

interface DocumentListProps {
  documents: DocumentInfo[];
  selectedDocId: string | null;
  onSelectDocument: (docId: string) => void;
  onDocumentDeleted: (docId: string) => void;
  isLoading: boolean;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  selectedDocId,
  onSelectDocument,
  onDocumentDeleted,
  isLoading,
}) => {
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleDelete = async (e: React.MouseEvent, docId: string) => {
    e.stopPropagation();
    if (!window.confirm('Delete this manual and all stored vectors/images?')) {
      return;
    }
    setDeletingId(docId);
    try {
      await deleteDocument(docId);
      onDocumentDeleted(docId);
    } catch (err) {
      console.error('Failed to delete document:', err);
      alert('Failed to delete document.');
    } finally {
      setDeletingId(null);
    }
  };

  const formatDate = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold text-slate-800 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-indigo-600" />
          Uploaded Manuals ({documents.length})
        </h2>
        {isLoading && <Loader2 className="w-4 h-4 text-indigo-600 animate-spin" />}
      </div>

      {documents.length === 0 ? (
        <div className="text-center py-8 text-slate-400 text-sm border-2 border-dashed border-slate-100 rounded-lg">
          No manuals uploaded yet. Upload a technical manual above to begin.
        </div>
      ) : (
        <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
          {documents.map((doc) => {
            const isSelected = doc.document_id === selectedDocId;
            const isDeleting = deletingId === doc.document_id;

            return (
              <div
                key={doc.document_id}
                onClick={() => doc.status === 'ready' && onSelectDocument(doc.document_id)}
                className={`p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between ${
                  isSelected
                    ? 'border-indigo-500 bg-indigo-50/50 ring-1 ring-indigo-500'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                } ${doc.status !== 'ready' ? 'opacity-70 cursor-not-allowed' : ''}`}
              >
                <div className="flex-1 min-w-0 mr-3">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm text-slate-800 truncate" title={doc.filename}>
                      {doc.filename}
                    </span>
                    <StatusBadge status={doc.status} />
                  </div>

                  <div className="flex items-center gap-4 text-xs text-slate-500 mt-1">
                    {doc.page_count !== null && (
                      <span className="flex items-center gap-1">
                        <Layers className="w-3.5 h-3.5 text-slate-400" />
                        {doc.page_count} pages
                      </span>
                    )}
                    <span className="flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5 text-slate-400" />
                      {formatDate(doc.upload_time)}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => handleDelete(e, doc.document_id)}
                    disabled={isDeleting}
                    className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-md transition-colors"
                    title="Delete manual"
                  >
                    {isDeleting ? (
                      <Loader2 className="w-4 h-4 animate-spin text-rose-500" />
                    ) : (
                      <Trash2 className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
