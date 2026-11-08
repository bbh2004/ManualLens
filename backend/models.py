from typing import Optional
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str

class StatusResponse(BaseModel):
    document_id: str
    status: str
    stage: Optional[str] = None
    error_message: Optional[str] = None

class QueryRequest(BaseModel):
    document_id: str
    question: str

class Citation(BaseModel):
    document_id: str
    page_number: int
    chunk_text: str
    source: str = Field(description="'semantic' or 'keyword'")

class RelevantPage(BaseModel):
    document_id: str
    page_number: int
    image_url: str
    visual_score: float

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    relevant_pages: list[RelevantPage] = []
    uncertainties: list[str] = []

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: Optional[int] = None
    upload_time: str
    status: str

class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]

class DeleteResponse(BaseModel):
    message: str

