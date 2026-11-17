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
        "srsly",
    )
)

@app.cls(
    image=colpali_image,
    gpu=GPU_TYPE,
    timeout=600,
    scaledown_window=300,
)
class ColPaliChunkEmbedder:
    """
    Parallel worker for embedding a partition chunk of pages in multi-container setups.
    Loads and caches vidore/colpali-v1.2 in GPU memory on container startup.
    """

    @modal.enter()
    def initialize(self):
        import torch
        from byaldi import RAGMultiModalModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ColPaliChunkEmbedder] Loading base ColPali model on {self.device} ({GPU_TYPE})...")
        self.base_model = RAGMultiModalModel.from_pretrained(
            "vidore/colpali-v1.2",
            index_root="/tmp",
            device=self.device,
            verbose=0,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ColPaliChunkEmbedder] Ready.")

    @modal.method()
    def ping(self) -> str:
        return "pong"

    @modal.method()
    def embed_chunk(self, chunk_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Processes a chunk of pages.
        chunk_data is {'items': [{'doc_id': int, 'filename': str, 'image_bytes': bytes}], 'batch_size': int}.
        Returns list of {'doc_id': int, 'filename': str, 'embedding': torch.Tensor (on cpu)}.
        """
        import io
        import torch
        from PIL import Image

        items = chunk_data.get("items", [])
        batch_size = chunk_data.get("batch_size", 2)
        inner = self.base_model.model
        results: list[dict[str, Any]] = []

        print(f"[ColPaliChunkEmbedder] Processing chunk of {len(items)} pages with batch_size={batch_size}...")

        for i in range(0, len(items), batch_size):
            sub = items[i : i + batch_size]
            pil_images = [Image.open(io.BytesIO(x["image_bytes"])).convert("RGB") for x in sub]
            with torch.inference_mode():
                batch = inner.processor.process_images(pil_images)
                batch = {
                    k: v.to(inner.device).to(
                        inner.model.dtype
                        if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                        else v.dtype
                    )
                    for k, v in batch.items()
                }
                embeddings = inner.model(**batch)
                unbound = list(torch.unbind(embeddings.to("cpu")))

            for x, emb in zip(sub, unbound):
                results.append({
                    "doc_id": x["doc_id"],
                    "filename": x["filename"],
                    "embedding": emb,
                })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[ColPaliChunkEmbedder] Finished processing chunk of {len(items)} pages.")
        return results


@app.cls(
    image=colpali_image,
    gpu=GPU_TYPE,
    volumes={INDEX_ROOT: volume},
    timeout=900,
    scaledown_window=300,
)
class ColPaliWorker:
    """
    Primary serverless GPU worker hosting vidore/colpali-v1.2 on Modal.
    Coordinates batched and parallel visual indexing, persists Byaldi-compatible
    indices in modal.Volume, and performs late-interaction multi-vector search.
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
    def index_pages(
        self,
        document_id: str,
        image_zip_bytes: bytes,
        batch_size: int = 2,
        parallel: bool = True,
        parallel_threshold: int = 50,
        max_containers: int = 4,
    ) -> dict[str, Any]:
        """
        Builds a ColPali late-interaction index from rendered page PNGs in image_zip_bytes.
        Supports single-container batched GPU indexing or horizontal multi-container
        parallel indexing via ColPaliChunkEmbedder.
        Index is stored persistently in the modal.Volume at INDEX_ROOT in Byaldi format.
        """
        import time
        import torch
        import srsly
        from PIL import Image

        t0 = time.time()
        temp_dir = Path(f"/tmp/pages_{document_id}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(image_zip_bytes), "r") as zf:
            zf.extractall(temp_dir)

        # Gather extracted png files and sort numerically by page number
        raw_png_files = list(temp_dir.rglob("*.png"))
        if not raw_png_files:
            raise ValueError(f"No PNG page images found in payload for document '{document_id}'")

        def _page_num_key(p: Path) -> int:
            m = re.search(r"page_(\d+)\.png", p.name)
            return int(m.group(1)) if m else 0

        png_files = sorted(raw_png_files, key=_page_num_key)
        total_pages = len(png_files)
        print(f"[ColPaliWorker] Starting indexing for '{document_id}' ({total_pages} pages). "
              f"batch_size={batch_size}, parallel={parallel}, threshold={parallel_threshold}, max_containers={max_containers}")

        index_name = f"vis_{document_id}"
        all_embeddings: list[torch.Tensor] = []
        all_filenames: list[str] = [p.name for p in png_files]
        containers_used = 1
        parallel_used = False

        if parallel and total_pages > parallel_threshold:
            parallel_used = True
            # Compute number of containers to scale out
            num_containers = min(max_containers, max(1, (total_pages + 49) // 50))
            containers_used = num_containers
            chunk_size = (total_pages + num_containers - 1) // num_containers

            print(f"[ColPaliWorker] Scaling out horizontally: {total_pages} pages -> "
                  f"{num_containers} workers (~{chunk_size} pages each).")

            # Form items with in-memory image bytes
            all_items: list[dict[str, Any]] = []
            for idx, p in enumerate(png_files):
                with open(p, "rb") as f:
                    all_items.append({
                        "doc_id": idx,
                        "filename": p.name,
                        "image_bytes": f.read(),
                    })

            chunks = [
                {
                    "items": all_items[i : i + chunk_size],
                    "batch_size": batch_size,
                }
                for i in range(0, total_pages, chunk_size)
            ]

            embedder = ColPaliChunkEmbedder()
            chunk_results = list(embedder.embed_chunk.map(chunks))

            flat_results = [item for sub in chunk_results for item in sub]
            flat_results.sort(key=lambda x: x["doc_id"])
            all_embeddings = [x["embedding"] for x in flat_results]
        else:
            # Single-container batched GPU indexing
            print(f"[ColPaliWorker] Running single-container batched indexing (batch_size={batch_size})...")
            inner = self.base_model.model
            for i in range(0, total_pages, batch_size):
                sub_paths = png_files[i : i + batch_size]
                pil_images = [Image.open(p).convert("RGB") for p in sub_paths]
                with torch.inference_mode():
                    batch = inner.processor.process_images(pil_images)
                    batch = {
                        k: v.to(inner.device).to(
                            inner.model.dtype
                            if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                            else v.dtype
                        )
                        for k, v in batch.items()
                    }
                    embeddings = inner.model(**batch)
                    unbound = list(torch.unbind(embeddings.to("cpu")))
                    all_embeddings.extend(unbound)

        # Write index in standard Byaldi format
        target_dir = Path(INDEX_ROOT) / index_name
        target_dir.mkdir(parents=True, exist_ok=True)
        emb_dir = target_dir / "embeddings"
        emb_dir.mkdir(exist_ok=True)

        chunk_size_export = 500
        for i in range(0, len(all_embeddings), chunk_size_export):
            torch.save(all_embeddings[i : i + chunk_size_export], emb_dir / f"embeddings_{i}.pt")

        embed_id_to_doc_id = {i: {"doc_id": i, "page_id": i + 1} for i in range(len(all_embeddings))}
        doc_ids_to_file_names = {i: all_filenames[i] for i in range(len(all_embeddings))}
        index_config = {
            "model_name": "vidore/colpali-v1.2",
            "full_document_collection": False,
            "highest_doc_id": len(all_embeddings) - 1,
            "resize_stored_images": False,
            "max_image_width": None,
            "max_image_height": None,
            "library_version": "0.0.7",
        }

        srsly.write_gzip_json(target_dir / "index_config.json.gz", index_config)
        srsly.write_gzip_json(target_dir / "embed_id_to_doc_id.json.gz", embed_id_to_doc_id)
        srsly.write_gzip_json(target_dir / "doc_ids_to_file_names.json.gz", doc_ids_to_file_names)
        srsly.write_gzip_json(target_dir / "metadata.json.gz", {})

        # Commit changes to persistent modal.Volume
        volume.commit()
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Clear any cached search index so the newly built index is reloaded fresh
        if index_name in self.index_cache:
            del self.index_cache[index_name]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        per_page = round(elapsed / total_pages, 3) if total_pages > 0 else 0.0
        print(f"[ColPaliWorker] Completed indexing {total_pages} pages in {elapsed:.2f}s "
              f"({per_page}s/page, parallel={parallel_used}, containers={containers_used}).")

        return {
            "status": "success",
            "document_id": document_id,
            "page_count": total_pages,
            "indexing_time_sec": round(elapsed, 2),
            "per_page_sec": per_page,
            "parallel_used": parallel_used,
            "containers_used": containers_used,
            "batch_size": batch_size,
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
