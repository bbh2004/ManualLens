import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional
from backend.config import SQLITE_DB_PATH
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.database")

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initializes the SQLite database schema if not present."""
    logger.info("Initializing SQLite database at %s", SQLITE_DB_PATH)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id     TEXT PRIMARY KEY,
                filename        TEXT NOT NULL,
                file_path       TEXT NOT NULL,
                page_count      INTEGER,
                upload_time     TEXT NOT NULL,
                status          TEXT NOT NULL,
                status_stage    TEXT,
                error_message   TEXT,
                file_size_bytes INTEGER
            );
            """
        )
        conn.commit()

def create_document(
    document_id: str,
    filename: str,
    file_path: str,
    file_size_bytes: int,
    initial_status: str = "processing",
    initial_stage: str = "extracting_text",
) -> dict[str, Any]:
    """Inserts a new document record into SQLite."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO documents (
                document_id, filename, file_path, page_count, upload_time,
                status, status_stage, error_message, file_size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                filename,
                file_path,
                None,
                now_iso,
                initial_status,
                initial_stage,
                None,
                file_size_bytes,
            ),
        )
        conn.commit()
    logger.info("Created document record: %s (%s)", document_id, filename)
    return {
        "document_id": document_id,
        "filename": filename,
        "status": initial_status,
        "stage": initial_stage,
    }

def update_status(
    document_id: str,
    status: str,
    stage: Optional[str] = None,
    error_message: Optional[str] = None,
    page_count: Optional[int] = None,
) -> None:
    """Updates document processing status, stage, error message, and page count."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "UPDATE documents SET status = ?"
        params: list[Any] = [status]

        if stage is not None:
            query += ", status_stage = ?"
            params.append(stage)
        if error_message is not None:
            query += ", error_message = ?"
            params.append(error_message)
        if page_count is not None:
            query += ", page_count = ?"
            params.append(page_count)

        query += " WHERE document_id = ?"
        params.append(document_id)

        cursor.execute(query, params)
        conn.commit()
    logger.info(
        "Updated document %s status -> %s (stage: %s, page_count: %s)",
        document_id,
        status,
        stage,
        page_count,
    )

def get_document(document_id: str) -> Optional[dict[str, Any]]:
    """Fetches a single document row by document_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

def list_documents() -> list[dict[str, Any]]:
    """Returns all documents ordered by upload_time descending."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY upload_time DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def delete_document(document_id: str) -> bool:
    """Deletes a document record from SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Deleted document record from DB: %s", document_id)
    return deleted

