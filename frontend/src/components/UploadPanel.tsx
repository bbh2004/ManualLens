import React, { useState, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { uploadDocument, getStatus } from '../api/client';
import { StatusBadge } from './StatusBadge';

interface UploadPanelProps {
  onUploadSuccess: (documentId: string) => void;
}

export const UploadPanel: React.FC<UploadPanelProps> = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [processingStatus, setProcessingStatus] = useState<'processing' | 'ready' | 'error' | null>(null);
  const [processingStage, setProcessingStage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const pollIntervalRef = useRef<number | null>(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setSelectedFile(acceptedFiles[0]);
        setErrorMessage(null);
        setProcessingStatus(null);
      }
    },
    disabled: isUploading || processingStatus === 'processing',
  });

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setErrorMessage(null);

    try {
      const res = await uploadDocument(selectedFile);
      setActiveDocId(res.document_id);
      setProcessingStatus('processing');
      setProcessingStage('extracting_text');
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to upload document.';
      setErrorMessage(msg);
      setProcessingStatus('error');
    } finally {
      setIsUploading(false);
    }
  };

  // Poll status endpoint every 2s
  useEffect(() => {
    if (!activeDocId || processingStatus !== 'processing') {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      return;
    }

    const checkStatus = async () => {
      try {
        const res = await getStatus(activeDocId);
        setProcessingStatus(res.status);
        setProcessingStage(res.stage || null);

        if (res.status === 'ready') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          onUploadSuccess(activeDocId);
        } else if (res.status === 'error') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setErrorMessage(res.error_message || 'Document ingestion failed.');
        }
      } catch (err: any) {
        logger_error(err);
      }
    };

    function logger_error(err: any) {
      console.warn('Status poll warning:', err);
    }

    pollIntervalRef.current = window.setInterval(checkStatus, 2000);
    checkStatus();

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [activeDocId, processingStatus, onUploadSuccess]);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
      <h2 className="text-base font-semibold text-slate-800 mb-3 flex items-center gap-2">
        <UploadCloud className="w-5 h-5 text-indigo-600" />
        Upload Technical Manual
      </h2>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-indigo-500 bg-indigo-50/50'
            : selectedFile
            ? 'border-emerald-300 bg-emerald-50/30'
            : 'border-slate-300 hover:border-slate-400 bg-slate-50/50'
        } ${isUploading || processingStatus === 'processing' ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center space-y-2">
          {selectedFile ? (
            <>
              <FileText className="w-8 h-8 text-emerald-600" />
              <div className="text-sm font-medium text-slate-700">{selectedFile.name}</div>
              <div className="text-xs text-slate-500">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </div>
            </>
          ) : (
            <>
              <UploadCloud className="w-8 h-8 text-slate-400" />
              <p className="text-sm text-slate-600">
                <span className="font-semibold text-indigo-600">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-slate-400">PDF manuals up to 50 MB</p>
            </>
          )}
        </div>
      </div>

      {selectedFile && !activeDocId && (
        <div className="mt-4 flex justify-end">
          <button
            onClick={handleUpload}
            disabled={isUploading}
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors shadow-sm"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Uploading...
              </>
            ) : (
              'Process Manual'
            )}
          </button>
        </div>
      )}

      {/* Progress & Status Indicator */}
      {processingStatus && (
        <div className="mt-4 p-3 bg-slate-50 rounded-lg border border-slate-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-slate-700 font-medium">
              {processingStatus === 'processing' && <Loader2 className="w-4 h-4 text-indigo-600 animate-spin" />}
              {processingStatus === 'ready' && <CheckCircle2 className="w-4 h-4 text-emerald-600" />}
              {processingStatus === 'error' && <AlertCircle className="w-4 h-4 text-rose-600" />}
              <span>Ingestion Status:</span>
            </div>
            <StatusBadge status={processingStatus} stage={processingStage || undefined} />
          </div>

          {processingStatus === 'processing' && (
            <div className="mt-2 text-xs text-slate-500 flex items-center justify-between">
              <span>Pipeline running (text extraction, BM25, embeddings, rendering)</span>
              <span className="font-mono">{processingStage}</span>
            </div>
          )}

          {processingStatus === 'ready' && (
            <div className="mt-2 text-xs text-emerald-700">
              Manual is indexed and ready for multimodal queries!
            </div>
          )}
        </div>
      )}

      {errorMessage && (
        <div className="mt-3 p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-700 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-500" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};

