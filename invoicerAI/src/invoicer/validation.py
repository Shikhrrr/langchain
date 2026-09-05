"""Business-level validation — Stage 4b of the pipeline.

Pydantic (models.py) validates *structure*: types, required fields, ranges.
This module validates *business rules*: whether the extracted values make sense
as a receipt — e.g. do the item totals add up to the subtotal?

Design:
- All checks return a list of human-readable violation strings.
- An empty list means the receipt passed all checks.
- The pipeline uses these violations to decide whether to mark a receipt
  as COMPLETED or NEEDS_REVIEW.
- Monetary comparisons use a configurable tolerance (default ±$0.05) to
  accommodate rounding differences common on printed receipts.
"""

import logging
from decimal import Decimal
from typing import List

from invoicer import config
from invoicer.models import Receipt

logger = logging.getLogger(__name__)

_TOLERANCE = Decimal(str(config.MONETARY_TOLERANCE))


def validate_business_rules(receipt: Receipt) -> List[str]:
    """Check business-level constraints on an extracted receipt.

    Args:
        receipt: A structurally valid Receipt instance (Pydantic already passed).

    Returns:
        List of violation descriptions. Empty list = all checks passed.
    """
    violations: List[str] = []

    # ------------------------------------------------------------------
    # 1. Individual item checks
    # ------------------------------------------------------------------
    for i, item in enumerate(receipt.items, start=1):
        if item.quantity <= 0:
            violations.append(f"Item {i} ({item.item_name!r}): quantity must be > 0, got {item.quantity}")
        if item.unit_price < Decimal("0"):
            violations.append(f"Item {i} ({item.item_name!r}): unit_price cannot be negative, got {item.unit_price}")
        if item.item_total is not None and item.item_total < Decimal("0"):
            violations.append(f"Item {i} ({item.item_name!r}): item_total cannot be negative, got {item.item_total}")

    # ------------------------------------------------------------------
    # 2. Top-level monetary checks
    # ------------------------------------------------------------------
    if receipt.tax is not None and receipt.tax < Decimal("0"):
        violations.append(f"tax cannot be negative, got {receipt.tax}")
    if receipt.discount is not None and receipt.discount < Decimal("0"):
        violations.append(f"discount cannot be negative, got {receipt.discount}")
    if receipt.total < Decimal("0"):
        violations.append(f"total cannot be negative, got {receipt.total}")

    # ------------------------------------------------------------------
    # 3. Subtotal cross-check (sum of item totals ≈ subtotal)
    # ------------------------------------------------------------------
    items_with_totals = [item for item in receipt.items if item.item_total is not None]
    if items_with_totals and receipt.subtotal is not None:
        computed_subtotal = sum(item.item_total for item in items_with_totals)  # type: ignore[misc]
        diff = abs(computed_subtotal - receipt.subtotal)
        if diff > _TOLERANCE:
            violations.append(
                f"Sum of item totals ({computed_subtotal}) does not match "
                f"subtotal ({receipt.subtotal}); difference {diff} exceeds tolerance {_TOLERANCE}"
            )

    # ------------------------------------------------------------------
    # 4. Total cross-check (subtotal + tax - discount ≈ total)
    # ------------------------------------------------------------------
    if receipt.subtotal is not None:
        tax = receipt.tax or Decimal("0")
        discount = receipt.discount or Decimal("0")
        expected_total = receipt.subtotal + tax - discount
        diff = abs(expected_total - receipt.total)
        if diff > _TOLERANCE:
            violations.append(
                f"subtotal ({receipt.subtotal}) + tax ({tax}) - discount ({discount}) "
                f"= {expected_total}, but total is {receipt.total}; "
                f"difference {diff} exceeds tolerance {_TOLERANCE}"
            )

    if violations:
        logger.warning("Business validation found %d violation(s):", len(violations))
        for v in violations:
            logger.warning("  • %s", v)
    else:
        logger.info("Business validation passed — no violations.")

    return violations
