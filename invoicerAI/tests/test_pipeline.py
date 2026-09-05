"""Pipeline integration tests (pipeline.py).

All external dependencies are mocked:
  - OCR (extract_text)
  - LLM extraction (extract_receipt)
  - Database persistence (save_receipt, create_tables, get_engine)

Tests cover the main success/failure paths including fallback and NEEDS_REVIEW.
"""

import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from invoicer.models import ProcessingStatus, Receipt, ReceiptItem


def _make_valid_receipt() -> Receipt:
    return Receipt(
        merchant="Test Store",
        items=[
            ReceiptItem(item_name="Widget", quantity=1, unit_price="9.99", item_total="9.99")
        ],
        subtotal="9.99",
        tax="0.80",
        total="10.79",
        status=ProcessingStatus.COMPLETED,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_image(tmp_path):
    """Create a tiny placeholder PNG so Path.exists() returns True."""
    from PIL import Image
    img_path = tmp_path / "receipt.png"
    img = Image.new("RGB", (100, 50), color=(255, 255, 255))
    img.save(img_path)
    return img_path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestPipelineSuccess:
    def test_successful_receipt_returns_completed(self, fake_image):
        from invoicer.pipeline import process_receipt

        with patch("invoicer.pipeline.preprocess_to_path", return_value=fake_image), \
             patch("invoicer.pipeline.extract_text", return_value="OCR text here"), \
             patch("invoicer.pipeline.extract_receipt", return_value=_make_valid_receipt()), \
             patch("invoicer.pipeline.validate_business_rules", return_value=[]), \
             patch("invoicer.pipeline._try_persist"):

            receipt = process_receipt(fake_image, persist=False)

        assert receipt.status == ProcessingStatus.COMPLETED
        assert len(receipt.items) == 1

    def test_result_contains_extracted_merchant(self, fake_image):
        from invoicer.pipeline import process_receipt

        with patch("invoicer.pipeline.preprocess_to_path", return_value=fake_image), \
             patch("invoicer.pipeline.extract_text", return_value="some text"), \
             patch("invoicer.pipeline.extract_receipt", return_value=_make_valid_receipt()), \
             patch("invoicer.pipeline.validate_business_rules", return_value=[]), \
             patch("invoicer.pipeline._try_persist"):

            receipt = process_receipt(fake_image, persist=False)

        assert receipt.merchant == "Test Store"


# ---------------------------------------------------------------------------
# OCR failure
# ---------------------------------------------------------------------------

class TestOCRFailure:
    def test_ocr_failure_returns_failed_status(self, fake_image):
        from invoicer.pipeline import process_receipt

        with patch("invoicer.pipeline.preprocess_to_path", return_value=fake_image), \
             patch("invoicer.pipeline.extract_text", side_effect=RuntimeError("All OCR providers failed")), \
             patch("invoicer.pipeline._try_persist"):

            receipt = process_receipt(fake_image, persist=False)

        assert receipt.status == ProcessingStatus.FAILED
        assert any("OCR" in e for e in receipt.validation_errors)

    def test_preprocessing_failure_falls_back_to_original(self, fake_image):
        """If preprocessing fails, the pipeline should still continue with the original image."""
        from invoicer.pipeline import process_receipt

        with patch("invoicer.pipeline.preprocess_to_path", side_effect=OSError("preprocess error")), \
             patch("invoicer.pipeline.extract_text", return_value="text from original"), \
             patch("invoicer.pipeline.extract_receipt", return_value=_make_valid_receipt()), \
             patch("invoicer.pipeline.validate_business_rules", return_value=[]), \
             patch("invoicer.pipeline._try_persist"):

            # Should not raise, should degrade gracefully
            receipt = process_receipt(fake_image, persist=False)

        assert receipt.status == ProcessingStatus.COMPLETED


# ---------------------------------------------------------------------------
# LLM extraction failure
# ---------------------------------------------------------------------------

class TestExtractionFailure:
    def test_extraction_none_returns_needs_review(self, fake_image):
        from invoicer.pipeline import process_receipt

        with patch("invoicer.pipeline.preprocess_to_path", return_value=fake_image), \
             patch("invoicer.pipeline.extract_text", return_value="some text"), \
             patch("invoicer.pipeline.extract_receipt", return_value=None), \
             patch("invoicer.pipeline._try_persist"):

            receipt = process_receipt(fake_image, persist=False)

        assert receipt.status == ProcessingStatus.NEEDS_REVIEW
        assert receipt.validation_errors  # should have an error message


# ---------------------------------------------------------------------------
# Business validation violations
# ---------------------------------------------------------------------------

class TestBusinessValidationViolations:
    def test_violations_result_in_needs_review(self, fake_image):
        from invoicer.pipeline import process_receipt

        with patch("invoicer.pipeline.preprocess_to_path", return_value=fake_image), \
             patch("invoicer.pipeline.extract_text", return_value="some text"), \
             patch("invoicer.pipeline.extract_receipt", return_value=_make_valid_receipt()), \
             patch("invoicer.pipeline.validate_business_rules", return_value=["total mismatch"]), \
             patch("invoicer.pipeline._try_persist"):

            receipt = process_receipt(fake_image, persist=False)

        assert receipt.status == ProcessingStatus.NEEDS_REVIEW
        assert "total mismatch" in receipt.validation_errors

    def test_zero_violations_result_in_completed(self, fake_image):
        from invoicer.pipeline import process_receipt

        with patch("invoicer.pipeline.preprocess_to_path", return_value=fake_image), \
             patch("invoicer.pipeline.extract_text", return_value="some text"), \
             patch("invoicer.pipeline.extract_receipt", return_value=_make_valid_receipt()), \
             patch("invoicer.pipeline.validate_business_rules", return_value=[]), \
             patch("invoicer.pipeline._try_persist"):

            receipt = process_receipt(fake_image, persist=False)

        assert receipt.status == ProcessingStatus.COMPLETED
