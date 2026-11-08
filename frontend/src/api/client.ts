import axios from 'axios';

// Base URL uses relative path to leverage Vite proxy to http://localhost:8000
const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface DocumentInfo {
  document_id: string;
  filename: string;
  page_count: number | null;
  upload_time: string;
  status: 'processing' | 'ready' | 'error';
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
}

export interface StatusResponse {
  document_id: string;
  status: 'processing' | 'ready' | 'error';
  stage?: string;
  error_message?: string;
}

export interface Citation {
  document_id: string;
  page_number: number;
  chunk_text: string;
  source: string;
}

export interface RelevantPage {
  document_id: string;
  page_number: number;
  image_url: string;
  visual_score: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  relevant_pages: RelevantPage[];
  uncertainties: string[];
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post<UploadResponse>('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function getStatus(documentId: string): Promise<StatusResponse> {
  const response = await apiClient.get<StatusResponse>(`/status/${documentId}`);
  return response.data;
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const response = await apiClient.get<{ documents: DocumentInfo[] }>('/documents');
  return response.data.documents;
}

export async function deleteDocument(documentId: string): Promise<{ message: string }> {
  const response = await apiClient.delete<{ message: string }>(`/documents/${documentId}`);
  return response.data;
}

export async function queryDocument(documentId: string, question: string): Promise<QueryResponse> {
  const response = await apiClient.post<QueryResponse>('/query', {
    document_id: documentId,
    question,
  });
  return response.data;
}

export function getPageUrl(documentId: string, pageNumber: number): string {
  return `/api/pages/${documentId}/${pageNumber}`;
}

