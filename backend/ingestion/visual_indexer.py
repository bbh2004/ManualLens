import io
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Any
from backend.config import QDRANT_DIR, USE_MODAL
from backend.utils.file_utils import get_page_image_dir
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.visual_indexer")

_visual_model: Optional[Any] = None

def get_byaldi_index_root() -> Path:
    """Returns persistent directory where byaldi stores late-interaction vector indexes locally."""
    index_root = QDRANT_DIR / "byaldi"
    index_root.mkdir(parents=True, exist_ok=True)
    return index_root

def get_visual_model():
    """Loads and caches the local ColPali multimodal model (CPU fallback)."""
    global _visual_model
    if _visual_model is None:
        import torch
        from byaldi import RAGMultiModalModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing local ColPali model (vidore/colpali-v1.2) on device: %s", device)
        _visual_model = RAGMultiModalModel.from_pretrained(
            "vidore/colpali-v1.2",
            index_root=str(get_byaldi_index_root()),
            device=device,
            verbose=1,
        )
    return _visual_model

def _pack_images_to_zip(image_paths: list[str]) -> bytes:
    """Packs rendered page images into an in-memory zip bytes buffer for network transport."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in image_paths:
            path_obj = Path(p)
            if path_obj.exists():
                zf.write(path_obj, arcname=path_obj.name)
    buffer.seek(0)
    return buffer.getvalue()

def build_visual_index(document_id: str, image_paths: list[str]) -> None:
    """
    Indexes rendered PDF page PNGs using ColPali multi-vector late-interaction embeddings.
    If USE_MODAL is enabled, offloads indexing to the serverless Modal GPU worker.
    Otherwise, executes indexing locally.
    """
    if not image_paths:
        logger.warning("No image paths provided for visual index of document %s", document_id)
        return

    index_name = f"vis_{document_id}"

    if USE_MODAL:
        logger.info(
            "Offloading visual indexing of %d pages for document '%s' to Modal GPU worker...",
            len(image_paths),
            document_id,
        )
        try:
            from backend.modal_worker import get_modal_worker
            worker = get_modal_worker()
            zip_bytes = _pack_images_to_zip(image_paths)
            result = worker.index_pages.remote(document_id, zip_bytes)
            logger.info("Modal GPU visual indexing succeeded for '%s': %s", document_id, result)
            return
        except Exception as e:
            logger.error("Modal visual indexing failed for '%s': %s", document_id, e, exc_info=True)
            raise

    # Local execution fallback
    img_dir = get_page_image_dir(document_id)
    logger.info("Building local ColPali visual index '%s' from directory %s...", index_name, img_dir)
    try:
        model = get_visual_model()
        model.index(
            input_path=str(img_dir),
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True,
        )
        logger.info("Local visual index '%s' successfully built and stored.", index_name)
    except Exception as e:
        logger.error("Failed to build local visual index '%s': %s", index_name, e, exc_info=True)
        raise

def delete_visual_index(document_id: str) -> None:
    """
    Removes the ColPali index from Modal cloud volume (if enabled)
    and from local disk storage.
    """
    if USE_MODAL:
        try:
            from backend.modal_worker import get_modal_worker
            worker = get_modal_worker()
            worker.delete_index.remote(document_id)
            logger.info("Deleted Modal cloud visual index for document '%s'", document_id)
        except Exception as e:
            logger.warning("Could not delete Modal visual index for '%s': %s", document_id, e)

    index_name = f"vis_{document_id}"
    index_root = get_byaldi_index_root()
    target_dir = index_root / index_name

    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)
            logger.info("Deleted local byaldi visual index directory: %s", target_dir)
        except Exception as e:
            logger.warning("Could not delete local byaldi index dir %s: %s", target_dir, e)

    default_byaldi_dir = Path(".byaldi") / index_name
    if default_byaldi_dir.exists():
        try:
            shutil.rmtree(default_byaldi_dir)
            logger.info("Deleted default .byaldi index directory: %s", default_byaldi_dir)
        except Exception as e:
            logger.warning("Could not delete default .byaldi dir: %s", e)
