"""Pipeline orchestration — the four-stage receipt processing flow.

Stage 1: Image preprocessing (Pillow)
Stage 2: OCR text extraction  (OCR.space → pytesseract fallback)
Stage 3: LLM extraction       (LangChain + local Llama 3)
Stage 4: Validation           (Pydantic structural + business rules)

Then:     Persistence          (PostgreSQL via SQLAlchemy)

Each stage is a discrete function. The pipeline logs its progress at every
stage and returns a Receipt with an appropriate ProcessingStatus:

  COMPLETED   — all stages passed, no business-rule violations
  NEEDS_REVIEW — extraction succeeded but business rules flagged issues,
                 OR extraction ultimately failed after all retries
  FAILED      — OCR failed completely (cannot proceed)

The caller receives a Receipt object regardless of outcome; inspect
receipt.status and receipt.validation_errors for details.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from invoicer import config
from invoicer.database import create_tables, get_engine, save_receipt
from invoicer.extraction import extract_receipt
from invoicer.models import ProcessingStatus, Receipt, ReceiptItem
from invoicer.ocr import extract_text
from invoicer.preprocessing import preprocess_to_path
from invoicer.validation import validate_business_rules

logger = logging.getLogger(__name__)


def _make_failed_receipt(validation_errors: list[str]) -> Receipt:
    """Return a minimal Receipt that represents a total pipeline failure."""
    return Receipt(
        items=[ReceiptItem(item_name="UNKNOWN", quantity=1, unit_price="0")],
        total="0",
        status=ProcessingStatus.FAILED,
        validation_errors=validation_errors,
    )


def process_receipt(
    image_path: str | Path,
    persist: bool = True,
    database_url: Optional[str] = None,
) -> Receipt:
    """Run the full 4-stage pipeline on a receipt image.

    Args:
        image_path:   Path to the input receipt image.
        persist:      If True, save result to PostgreSQL. Set False for testing.
        database_url: Override DATABASE_URL from config (useful in tests).

    Returns:
        Receipt with .status and .validation_errors populated.
    """
    image_path = Path(image_path)
    logger.info("=" * 60)
    logger.info("Pipeline started for: %s", image_path)
    logger.info("=" * 60)

    raw_ocr_text = ""

    # ------------------------------------------------------------------
    # Stage 1 — Image Preprocessing
    # ------------------------------------------------------------------
    logger.info("[Stage 1] Preprocessing image...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        preprocessed_path = preprocess_to_path(image_path, tmp_path)
        logger.info("[Stage 1] Preprocessing complete → %s", preprocessed_path)
    except Exception as e:
        logger.error("[Stage 1] Preprocessing failed: %s. Using original image.", e)
        preprocessed_path = image_path  # degrade gracefully

    # ------------------------------------------------------------------
    # Stage 2 — OCR
    # ------------------------------------------------------------------
    logger.info("[Stage 2] Running OCR...")
    try:
        raw_ocr_text = extract_text(preprocessed_path)
        logger.info("[Stage 2] OCR complete — %d characters extracted.", len(raw_ocr_text))
        logger.debug("[Stage 2] OCR text:\n%s", raw_ocr_text)
    except Exception as e:
        logger.error("[Stage 2] OCR failed completely: %s", e)
        receipt = _make_failed_receipt([f"OCR failure: {e}"])
        if persist:
            _try_persist(receipt, raw_ocr_text="", database_url=database_url)
        return receipt

    # ------------------------------------------------------------------
    # Stage 3 — LLM Extraction
    # ------------------------------------------------------------------
    logger.info("[Stage 3] Running LLM extraction (model=%s)...", config.OLLAMA_MODEL)
    extracted: Optional[Receipt] = extract_receipt(raw_ocr_text)

    if extracted is None:
        logger.error("[Stage 3] Extraction returned None after all attempts.")
        receipt = Receipt(
            items=[ReceiptItem(item_name="UNKNOWN", quantity=1, unit_price="0")],
            total="0",
            status=ProcessingStatus.NEEDS_REVIEW,
            validation_errors=["LLM extraction failed after all retry attempts."],
        )
        if persist:
            _try_persist(receipt, raw_ocr_text, database_url)
        return receipt

    logger.info("[Stage 3] Extraction complete — %d item(s) found.", len(extracted.items))

    # ------------------------------------------------------------------
    # Stage 4 — Business Validation
    # ------------------------------------------------------------------
    logger.info("[Stage 4] Running business validation...")
    violations = validate_business_rules(extracted)

    if violations:
        extracted.status = ProcessingStatus.NEEDS_REVIEW
        extracted.validation_errors = violations
        logger.warning("[Stage 4] Receipt flagged NEEDS_REVIEW — %d violation(s).", len(violations))
    else:
        extracted.status = ProcessingStatus.COMPLETED
        logger.info("[Stage 4] Validation passed — receipt is COMPLETED.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    if persist:
        _try_persist(extracted, raw_ocr_text, database_url)

    logger.info("Pipeline finished. Status: %s", extracted.status.value)
    return extracted


def _try_persist(receipt: Receipt, raw_ocr_text: str, database_url: Optional[str]) -> None:
    """Attempt to persist receipt to the database; log but don't crash on failure."""
    try:
        engine = get_engine(database_url)
        create_tables(engine)
        receipt_id = save_receipt(receipt, raw_ocr_text, engine)
        logger.info("Receipt persisted with id=%d.", receipt_id)
    except Exception as e:
        logger.error("Database persistence failed (receipt NOT saved): %s", e)
