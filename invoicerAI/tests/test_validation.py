"""Tests for business-level validation (validation.py).

These tests have zero external dependencies — no database, no LLM, no OCR.
They validate the pure logic of validate_business_rules().
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

# Make src importable without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from invoicer.models import Receipt, ReceiptItem, ProcessingStatus
from invoicer.validation import validate_business_rules


def _make_item(name="Apple", qty=2, unit_price="1.50", item_total=None):
    return ReceiptItem(
        item_name=name,
        quantity=qty,
        unit_price=unit_price,
        item_total=item_total,
    )


def _make_receipt(**kwargs):
    defaults = dict(
        merchant="Test Store",
        items=[_make_item()],
        total="3.00",
        status=ProcessingStatus.COMPLETED,
    )
    defaults.update(kwargs)
    return Receipt(**defaults)


# ---------------------------------------------------------------------------
# Valid receipt — no violations expected
# ---------------------------------------------------------------------------

class TestValidReceipt:
    def test_simple_receipt_passes(self):
        receipt = _make_receipt(
            items=[_make_item(qty=2, unit_price="1.50", item_total="3.00")],
            subtotal="3.00",
            tax="0.30",
            total="3.30",
        )
        assert validate_business_rules(receipt) == []

    def test_receipt_without_optional_fields_passes(self):
        # subtotal/tax/discount absent → cross-checks are skipped
        receipt = _make_receipt(total="5.00")
        assert validate_business_rules(receipt) == []

    def test_zero_tax_passes(self):
        receipt = _make_receipt(
            items=[_make_item(qty=1, unit_price="10.00", item_total="10.00")],
            subtotal="10.00",
            tax="0.00",
            total="10.00",
        )
        assert validate_business_rules(receipt) == []


# ---------------------------------------------------------------------------
# Negative value violations — caught by Pydantic structural validation
# ---------------------------------------------------------------------------
# Architecture note: Pydantic's ge=0 constraint catches negative monetary values
# at the structural layer before business validation ever runs.
# These tests verify that the structural layer is doing its job correctly.

class TestNegativeValues:
    def test_negative_unit_price_rejected_by_pydantic(self):
        """Pydantic (structural) rejects negative unit prices before business validation."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError, match="greater_than_equal"):
            ReceiptItem(item_name="Bad Item", quantity=1, unit_price="-5.00")

    def test_negative_total_rejected_by_pydantic(self):
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError, match="greater_than_equal"):
            Receipt(items=[_make_item()], total="-1.00")

    def test_negative_tax_rejected_by_pydantic(self):
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            Receipt(
                merchant="Test",
                items=[_make_item()],
                tax="-0.50",
                total="5.00",
            )

    def test_negative_discount_rejected_by_pydantic(self):
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            Receipt(
                merchant="Test",
                items=[_make_item()],
                discount="-1.00",
                total="5.00",
            )


# ---------------------------------------------------------------------------
# Subtotal cross-check
# ---------------------------------------------------------------------------

class TestSubtotalCrossCheck:
    def test_item_totals_match_subtotal(self):
        items = [
            _make_item("A", qty=2, unit_price="3.00", item_total="6.00"),
            _make_item("B", qty=1, unit_price="4.00", item_total="4.00"),
        ]
        receipt = _make_receipt(items=items, subtotal="10.00", total="10.00")
        assert validate_business_rules(receipt) == []

    def test_subtotal_mismatch_flagged(self):
        items = [_make_item(qty=2, unit_price="3.00", item_total="6.00")]
        # Subtotal says 9.00 but item total is 6.00
        receipt = _make_receipt(items=items, subtotal="9.00", total="9.00")
        violations = validate_business_rules(receipt)
        assert any("subtotal" in v.lower() for v in violations)

    def test_small_rounding_within_tolerance_passes(self):
        # Difference of $0.02 — within the default $0.05 tolerance
        items = [_make_item(qty=3, unit_price="1.67", item_total="5.01")]
        receipt = _make_receipt(items=items, subtotal="5.00", total="5.00")
        assert validate_business_rules(receipt) == []


# ---------------------------------------------------------------------------
# Total cross-check
# ---------------------------------------------------------------------------

class TestTotalCrossCheck:
    def test_total_matches_subtotal_plus_tax(self):
        items = [_make_item(qty=1, unit_price="10.00", item_total="10.00")]
        receipt = _make_receipt(
            items=items, subtotal="10.00", tax="1.00", total="11.00"
        )
        assert validate_business_rules(receipt) == []

    def test_total_with_discount_passes(self):
        items = [_make_item(qty=1, unit_price="20.00", item_total="20.00")]
        receipt = _make_receipt(
            items=items, subtotal="20.00", tax="2.00", discount="5.00", total="17.00"
        )
        assert validate_business_rules(receipt) == []

    def test_total_mismatch_flagged(self):
        items = [_make_item(qty=1, unit_price="10.00", item_total="10.00")]
        # subtotal 10 + tax 1 = 11, but total is 15 — clear error
        receipt = _make_receipt(items=items, subtotal="10.00", tax="1.00", total="15.00")
        violations = validate_business_rules(receipt)
        assert any("total" in v.lower() for v in violations)

    def test_large_rounding_exceeds_tolerance(self):
        # Difference of $0.10 — exceeds the $0.05 tolerance
        items = [_make_item(qty=1, unit_price="10.00", item_total="10.00")]
        receipt = _make_receipt(items=items, subtotal="10.00", tax="0.00", total="10.10")
        violations = validate_business_rules(receipt)
        assert len(violations) > 0
