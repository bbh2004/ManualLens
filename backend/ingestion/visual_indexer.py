import shutil
from pathlib import Path
from typing import Optional
import torch
from byaldi import RAGMultiModalModel
from backend.config import QDRANT_DIR
from backend.utils.file_utils import get_page_image_dir
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.visual_indexer")

_visual_model: Optional[RAGMultiModalModel] = None

def get_byaldi_index_root() -> Path:
    """Returns persistent directory where byaldi stores late-interaction vector indexes."""
    index_root = QDRANT_DIR / "byaldi"
    index_root.mkdir(parents=True, exist_ok=True)
    return index_root

def get_visual_model() -> RAGMultiModalModel:
    """Loads and caches the ColPali multimodal model."""
    global _visual_model
    if _visual_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing ColPali model (vidore/colpali-v1.2) on device: %s", device)
        _visual_model = RAGMultiModalModel.from_pretrained(
            "vidore/colpali-v1.2",
            index_root=str(get_byaldi_index_root()),
            device=device,
            verbose=1,
        )
    return _visual_model

def build_visual_index(document_id: str, image_paths: list[str]) -> None:
    """
    Indexes rendered PDF page PNGs using ColPali multi-vector late-interaction embeddings.
    Stored with index_name = f"vis_{document_id}".
    """
    if not image_paths:
        logger.warning("No image paths provided for visual index of document %s", document_id)
        return

    img_dir = get_page_image_dir(document_id)
    index_name = f"vis_{document_id}"
    logger.info("Building ColPali visual index '%s' from directory %s...", index_name, img_dir)

    try:
        model = get_visual_model()
        model.index(
            input_path=str(img_dir),
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True,
        )
        logger.info("Visual index '%s' successfully built and stored.", index_name)
    except Exception as e:
        logger.error("Failed to build visual index '%s': %s", index_name, e, exc_info=True)
        raise

def delete_visual_index(document_id: str) -> None:
    """Removes the ColPali index directory from disk for document_id."""
    index_name = f"vis_{document_id}"
    index_root = get_byaldi_index_root()
    target_dir = index_root / index_name

    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)
            logger.info("Deleted byaldi visual index directory: %s", target_dir)
        except Exception as e:
            logger.warning("Could not delete byaldi index dir %s: %s", target_dir, e)

    # Also clean .byaldi if created there
    default_byaldi_dir = Path(".byaldi") / index_name
    if default_byaldi_dir.exists():
        try:
            shutil.rmtree(default_byaldi_dir)
            logger.info("Deleted default .byaldi index directory: %s", default_byaldi_dir)
        except Exception as e:
            logger.warning("Could not delete default .byaldi dir: %s", e)

