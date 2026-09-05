"""OCR module — Stage 2 of the pipeline.

Design: provider abstraction with automatic fallback.

Primary:  OCR.space REST API  (cloud, high accuracy)
Fallback: pytesseract         (local, no API key required)

The pipeline calls ``extract_text()``. If the primary provider raises an
exception the fallback is tried automatically. This keeps the caller simple
while making the system resilient to transient API unavailability.
"""

import logging
from pathlib import Path
from typing import Union

import pytesseract
import requests
from PIL import Image

from invoicer import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary provider — OCR.space
# ---------------------------------------------------------------------------

def _ocr_space(image_path: Union[str, Path]) -> str:
    """Call the OCR.space REST API and return extracted text.

    Args:
        image_path: Path to the (preprocessed) image file.

    Returns:
        Extracted text string.

    Raises:
        ValueError: If OCR_SPACE_API_KEY is not configured.
        RuntimeError: If the API returns a non-success exit code.
        requests.RequestException: On network/timeout errors.
    """
    if not config.OCR_SPACE_API_KEY:
        raise ValueError(
            "OCR_SPACE_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://ocr.space/ocrapi"
        )

    logger.info("OCR: calling OCR.space API for %s", image_path)

    payload = {
        "apikey": config.OCR_SPACE_API_KEY,
        "language": config.OCR_LANGUAGE,
        "isOverlayRequired": False,
    }

    with open(image_path, "rb") as f:
        response = requests.post(
            config.OCR_SPACE_URL,
            files={"filename": f},
            data=payload,
            timeout=config.OCR_TIMEOUT,
        )
    response.raise_for_status()

    result = response.json()
    if result.get("OCRExitCode") != 1:
        error_msg = result.get("ErrorMessage", "Unknown OCR.space error")
        raise RuntimeError(f"OCR.space failed (exit code {result.get('OCRExitCode')}): {error_msg}")

    parsed_results = result.get("ParsedResults", [])
    if not parsed_results:
        raise RuntimeError("OCR.space returned no parsed results.")

    text = parsed_results[0].get("ParsedText", "")
    if not text.strip():
        raise RuntimeError("OCR.space returned empty text.")

    logger.info("OCR.space succeeded: extracted %d characters", len(text))
    return text


# ---------------------------------------------------------------------------
# Fallback provider — pytesseract (local)
# ---------------------------------------------------------------------------

def _pytesseract(image_path: Union[str, Path]) -> str:
    """Extract text using local Tesseract via pytesseract.

    Args:
        image_path: Path to the (preprocessed) image file.

    Returns:
        Extracted text string.

    Raises:
        RuntimeError: If Tesseract is not installed or returns empty text.
    """
    logger.info("OCR: falling back to pytesseract for %s", image_path)

    try:
        img = Image.open(image_path)
        # PSM 6: assume a uniform block of text (good for receipts)
        text = pytesseract.image_to_string(img, config="--psm 6")
    except pytesseract.pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract is not installed. Install it with: "
            "  macOS: brew install tesseract\n"
            "  Ubuntu: sudo apt-get install tesseract-ocr"
        )

    if not text.strip():
        raise RuntimeError("pytesseract returned empty text.")

    logger.info("pytesseract succeeded: extracted %d characters", len(text))
    return text


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_text(image_path: Union[str, Path]) -> str:
    """Extract text from a receipt image, falling back to Tesseract if needed.

    Tries OCR.space first. On any exception, logs a warning and tries
    pytesseract. If both fail, raises RuntimeError.

    Args:
        image_path: Path to the preprocessed receipt image.

    Returns:
        Raw OCR text.

    Raises:
        RuntimeError: If both providers fail.
    """
    errors = []

    # --- Primary: OCR.space ---
    try:
        return _ocr_space(image_path)
    except Exception as e:
        logger.warning("OCR.space failed: %s — attempting pytesseract fallback.", e)
        errors.append(f"OCR.space: {e}")

    # --- Fallback: pytesseract ---
    try:
        return _pytesseract(image_path)
    except Exception as e:
        logger.error("pytesseract fallback also failed: %s", e)
        errors.append(f"pytesseract: {e}")

    raise RuntimeError(
        "All OCR providers failed:\n" + "\n".join(f"  • {err}" for err in errors)
    )
