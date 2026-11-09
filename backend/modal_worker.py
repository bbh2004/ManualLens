import io
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Optional
import modal

APP_NAME = os.getenv("MODAL_APP_NAME", "manuallens-colpali")
VOLUME_NAME = "manuallens-byaldi-volume"
INDEX_ROOT = "/root/byaldi_indices"
GPU_TYPE = os.getenv("MODAL_GPU", "T4")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

colpali_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "torch",
        "torchvision",
        "byaldi>=0.0.6",
        "colpali-engine>=0.3.0",
        "transformers>=4.45.0",
        "accelerate",
        "pillow",
        "qdrant-client",
    )
)

@app.cls(
    image=colpali_image,
    gpu=GPU_TYPE,
    volumes={INDEX_ROOT: volume},
    timeout=600,
    scaledown_window=300,
)
class ColPaliWorker:
    """
    Serverless GPU worker hosting vidore/colpali-v1.2 on Modal.
    Offloads heavy visual indexing and late-interaction multi-vector search
    from local CPU to cloud GPU.
    """

    @modal.enter()
    def initialize(self):
        """Initializes and caches the base ColPali model in GPU memory on container startup."""
        import torch
        from byaldi import RAGMultiModalModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ColPaliWorker] Initializing base ColPali model on device: {self.device} ({GPU_TYPE})")
        self.base_model = RAGMultiModalModel.from_pretrained(
            "vidore/colpali-v1.2",
            index_root=INDEX_ROOT,
            device=self.device,
            verbose=0,
        )
        self.index_cache: dict[str, RAGMultiModalModel] = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ColPaliWorker] Base model initialized and ready in GPU memory.")

    @modal.method()
    def ping(self) -> str:
        """Health check method to verify worker connectivity and warm up container."""
        return "pong"

    @modal.method()
    def index_pages(self, document_id: str, image_zip_bytes: bytes) -> dict[str, Any]:
        """
        Extracts rendered page PNGs from zip bytes and builds a ColPali late-interaction
        index. Index is stored persistently in the modal.Volume at INDEX_ROOT.
        """
        import time
        import torch

        t0 = time.time()
        temp_dir = Path(f"/tmp/pages_{document_id}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(image_zip_bytes), "r") as zf:
            zf.extractall(temp_dir)

        # Gather extracted png files
        png_files = list(temp_dir.rglob("*.png"))
        print(f"[ColPaliWorker] Indexing {len(png_files)} pages for document '{document_id}'...")

        index_name = f"vis_{document_id}"

        # Reset in-memory index structures on base model to allow multi-document indexing on the same worker
        inner = getattr(self.base_model, "model", None)
        if inner is not None:
            inner.index_name = index_name
            inner.collection = {}
            inner.indexed_embeddings = []
            inner.embed_id_to_doc_id = {}
            inner.doc_id_to_metadata = {}
            inner.doc_ids_to_file_names = {}
            inner.doc_ids = set()
            inner.highest_doc_id = -1

        self.base_model.index(
            input_path=str(temp_dir),
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True,
        )

        # Commit changes to persistent modal.Volume
        volume.commit()
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Clear any cached search index so the newly built index is reloaded fresh
        if index_name in self.index_cache:
            del self.index_cache[index_name]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"[ColPaliWorker] Completed indexing {len(png_files)} pages in {elapsed:.2f}s.")
        return {
            "status": "success",
            "document_id": document_id,
            "page_count": len(png_files),
            "indexing_time_sec": round(elapsed, 2),
        }

    @modal.method()
    def search_pages(self, document_id: str, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Searches the visual index for document_id using the query question.
        Returns top-k ranked candidate pages and their scores.
        """
        import torch
        from byaldi import RAGMultiModalModel

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        index_name = f"vis_{document_id}"
        target_dir = Path(INDEX_ROOT) / index_name

        volume.reload()
        if not target_dir.exists():
            print(f"[ColPaliWorker] Index {index_name} not found in {INDEX_ROOT}")
            return []

        if index_name not in self.index_cache:
            print(f"[ColPaliWorker] Loading visual index '{index_name}' from {INDEX_ROOT}...")
            self.index_cache[index_name] = RAGMultiModalModel.from_index(
                index_path=index_name,
                index_root=INDEX_ROOT,
                device=self.device,
                verbose=0,
            )

        model = self.index_cache[index_name]
        with torch.inference_mode():
            raw_results = model.search(question, k=top_k)

        if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], list):
            results_list = raw_results[0]
        else:
            results_list = raw_results

        output: list[dict[str, Any]] = []
        doc_names = getattr(model.model, "doc_ids_to_file_names", {}) or {}

        for r in results_list:
            doc_id_val = getattr(r, "doc_id", None)
            score = getattr(r, "score", 0.0)
            page_num = getattr(r, "page_num", 1)

            if doc_id_val is not None and doc_id_val in doc_names:
                fname = str(doc_names[doc_id_val])
                m = re.search(r"page_(\d+)\.png", fname)
                if m:
                    page_num = int(m.group(1))

            output.append({
                "page_number": int(page_num),
                "visual_score": round(float(score), 4),
            })

        print(f"[ColPaliWorker] Found {len(output)} candidate pages for document '{document_id}'.")
        return output

    @modal.method()
    def delete_index(self, document_id: str) -> bool:
        """Removes the visual index for document_id from the volume."""
        import torch

        index_name = f"vis_{document_id}"
        target_dir = Path(INDEX_ROOT) / index_name

        volume.reload()
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
            volume.commit()
            if index_name in self.index_cache:
                del self.index_cache[index_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[ColPaliWorker] Successfully deleted index '{index_name}' from volume.")
            return True
        return False


_cached_worker_client: Optional[Any] = None

def get_modal_worker():
    """
    Returns a cached client connection to the deployed Modal ColPaliWorker.
    """
    global _cached_worker_client
    if _cached_worker_client is None:
        try:
            cls = modal.Cls.from_name(APP_NAME, "ColPaliWorker")
            _cached_worker_client = cls()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to deployed Modal app '{APP_NAME}': {e}. "
                "Ensure you have deployed the worker using 'modal deploy backend/modal_worker.py'."
            ) from e
    return _cached_worker_client
