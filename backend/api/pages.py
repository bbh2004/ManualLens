from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from backend.config import PAGE_IMAGES_DIR
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.api.pages")
router = APIRouter(prefix="/api", tags=["pages"])

@router.get("/pages/{document_id}/{page_number}")
def get_page_image(document_id: str, page_number: int):
    """
    Serves a rendered PNG image of the specified page from storage/page_images/{document_id}/page_{n}.png.
    """
    img_path = PAGE_IMAGES_DIR / document_id / f"page_{page_number}.png"
    if not img_path.exists():
        logger.warning("Requested page image not found: %s", img_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page image {page_number} for document '{document_id}' does not exist.",
        )

    return FileResponse(
        path=str(img_path),
        media_type="image/png",
        filename=f"{document_id}_page_{page_number}.png",
    )

