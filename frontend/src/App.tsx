import { useState, useEffect, useCallback } from 'react';
import { Layers, ShieldCheck, Sparkles } from 'lucide-react';
import type { DocumentInfo } from './api/client';
import { listDocuments } from './api/client';
import { UploadPanel } from './components/UploadPanel';
import { DocumentList } from './components/DocumentList';
import { ChatInterface } from './components/ChatInterface';
import { PageViewer } from './components/PageViewer';

export function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [inspectingPage, setInspectingPage] = useState<number | null>(null);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);

  const fetchDocuments = useCallback(async () => {
    setIsLoadingDocs(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);

      // Auto-select first ready document if none currently selected
      setSelectedDocId((prev) => {
        if (!prev && docs.length > 0) {
          const firstReady = docs.find((d) => d.status === 'ready');
          return firstReady ? firstReady.document_id : null;
        }
        return prev;
      });
    } catch (err) {
      console.error('Failed to load documents list:', err);
    } finally {
      setIsLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    (async () => {
      setIsLoadingDocs(true);
      try {
        const docs = await listDocuments();
        if (!ignore) {
          setDocuments(docs);
          setSelectedDocId((prev) => {
            if (!prev && docs.length > 0) {
              const firstReady = docs.find((d) => d.status === 'ready');
              return firstReady ? firstReady.document_id : null;
            }
            return prev;
          });
        }
      } catch (err) {
        console.error('Failed to load documents list:', err);
      } finally {
        if (!ignore) setIsLoadingDocs(false);
      }
    })();
    return () => {
      ignore = true;
    };
  }, []);

  const handleUploadSuccess = (docId: string) => {
    setSelectedDocId(docId);
    fetchDocuments();
  };

  const handleDocumentDeleted = (deletedId: string) => {
    if (selectedDocId === deletedId) {
      setSelectedDocId(null);
    }
    fetchDocuments();
  };

  const activeDoc = documents.find((d) => d.document_id === selectedDocId);

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-600 flex items-center justify-center shadow-md shadow-indigo-100">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold text-slate-900 tracking-tight">ManualLens</h1>
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                  Multimodal RAG
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Hybrid semantic search & visual page retrieval for technical manuals
              </p>
            </div>
          </div>

          <div className="hidden sm:flex items-center gap-4 text-xs text-slate-600">
            <span className="flex items-center gap-1.5 font-medium">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              Grounded Citations
            </span>
            <span className="flex items-center gap-1.5 font-medium">
              <Sparkles className="w-4 h-4 text-indigo-600" />
              Llama 3.1 & MiniLM
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Upload & Document Management */}
          <div className="lg:col-span-5 space-y-6">
            <UploadPanel onUploadSuccess={handleUploadSuccess} />
            <DocumentList
              documents={documents}
              selectedDocId={selectedDocId}
              onSelectDocument={(id) => setSelectedDocId(id)}
              onDocumentDeleted={handleDocumentDeleted}
              isLoading={isLoadingDocs}
            />
          </div>

          {/* Right Column: Q&A Chat Interface */}
          <div className="lg:col-span-7">
            <ChatInterface
              activeDocId={selectedDocId}
              activeDocName={activeDoc ? activeDoc.filename : null}
              onOpenPage={(pageNum) => setInspectingPage(pageNum)}
            />
          </div>
        </div>
      </main>

      {/* Full-size Page Inspector Lightbox Modal */}
      {selectedDocId && (
        <PageViewer
          documentId={selectedDocId}
          pageNumber={inspectingPage}
          onClose={() => setInspectingPage(null)}
        />
      )}
    </div>
  );
}

export default App;
