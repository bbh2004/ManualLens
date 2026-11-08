from PIL import Image
import pytesseract
from backend.utils.logging_config import get_logger

logger = get_logger("manuallens.ocr")

def run_ocr(image: Image.Image) -> str:
    """
    Applies OCR using pytesseract on a rendered page image.
    Gracefully catches and logs errors if the tesseract system binary is not found.
    """
    try:
        text = pytesseract.image_to_string(image)
        logger.debug("OCR extracted %d characters", len(text))
        return text
    except pytesseract.TesseractNotFoundError:
        logger.warning(
            "Tesseract binary not found on system PATH. OCR fallback skipped."
        )
        return ""
    except Exception as e:
        logger.warning("OCR processing error: %s", e)
        return ""

