import React, { useState } from 'react';
import { Send, MessageSquare, Loader2, HelpCircle } from 'lucide-react';
import type { QueryResponse } from '../api/client';
import { queryDocument } from '../api/client';
import { AnswerCard } from './AnswerCard';

interface ChatInterfaceProps {
  activeDocId: string | null;
  activeDocName: string | null;
  onOpenPage: (pageNumber: number) => void;
}

interface QnAPair {
  question: string;
  response: QueryResponse;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  activeDocId,
  activeDocName,
  onOpenPage,
}) => {
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [history, setHistory] = useState<QnAPair[]>([]);

  const handleAsk = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!question.trim() || !activeDocId || isLoading) return;

    const currentQ = question.trim();
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const res = await queryDocument(activeDocId, currentQ);
      setHistory((prev) => [{ question: currentQ, response: res }, ...prev]);
      setQuestion('');
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to answer query.';
      setErrorMsg(msg);
    } finally {
      setIsLoading(false);
    }
  };

  if (!activeDocId) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center">
        <HelpCircle className="w-12 h-12 text-slate-300 mx-auto mb-3" />
        <h3 className="text-base font-semibold text-slate-700">No Manual Selected</h3>
        <p className="text-sm text-slate-500 max-w-sm mx-auto mt-1">
          Select a processed manual from the list on the left to ask questions and inspect citations.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Search Input Box */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-indigo-600" />
            <h2 className="text-base font-semibold text-slate-800">
              Query Manual
            </h2>
          </div>
          <span className="text-xs bg-indigo-50 text-indigo-700 font-medium px-2.5 py-1 rounded-md border border-indigo-100 max-w-[240px] truncate">
            {activeDocName}
          </span>
        </div>

        <form onSubmit={handleAsk} className="relative">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleAsk();
              }
            }}
            placeholder="Ask a question (e.g. 'What does error E17 mean and how do I fix it?', 'Show me the pinout diagram')..."
            rows={3}
            disabled={isLoading}
            className="w-full text-sm rounded-lg border border-slate-300 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 p-3 pr-24 resize-none transition-colors"
          />

          <div className="absolute right-3 bottom-3 flex items-center gap-2">
            <button
              type="submit"
              disabled={isLoading || !question.trim()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 text-white text-xs font-semibold rounded-md hover:bg-indigo-700 disabled:opacity-40 transition-colors shadow-sm"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Retrieving...
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  Ask
                </>
              )}
            </button>
          </div>
        </form>

        {errorMsg && (
          <div className="mt-3 p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-700">
            {errorMsg}
          </div>
        )}
      </div>

      {/* Answers Stream */}
      <div className="space-y-5">
        {history.map((item, index) => (
          <AnswerCard
            key={index}
            question={item.question}
            response={item.response}
            documentId={activeDocId}
            onOpenPage={onOpenPage}
          />
        ))}
      </div>
    </div>
  );
};
