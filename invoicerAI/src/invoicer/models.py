"""Pydantic models for structured receipt data.

Key design choices:
- Decimal (not float) for all monetary values — avoids floating-point rounding errors.
- Pydantic validators enforce structural correctness (type, range, presence).
- Business-rule validation (e.g. total ≈ subtotal + tax - discount) lives in validation.py.
- ProcessingStatus tracks the pipeline outcome and is persisted alongside the receipt.
"""

from datetime import date as DateType
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ProcessingStatus(str, Enum):
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"


class ReceiptItem(BaseModel):
    """A single line-item on a receipt."""

    item_name: str = Field(description="The name or description of the purchased item.")
    quantity: int = Field(gt=0, description="Number of units purchased (must be positive).")
    unit_price: Decimal = Field(ge=Decimal("0"), description="Price per single unit.")
    item_total: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Total for this line (quantity × unit_price). May be absent on some receipts.",
    )

    @field_validator("unit_price", "item_total", mode="before")
    @classmethod
    def coerce_to_decimal(cls, v):
        """Accept numeric strings and floats; convert to Decimal."""
        if v is None:
            return v
        try:
            return Decimal(str(v))
        except Exception:
            raise ValueError(f"Cannot convert {v!r} to Decimal")

    @model_validator(mode="after")
    def validate_item_total(self) -> "ReceiptItem":
        """If item_total is provided, it should roughly equal quantity × unit_price."""
        if self.item_total is not None:
            expected = Decimal(str(self.quantity)) * self.unit_price
            diff = abs(self.item_total - expected)
            if diff > Decimal("0.10"):
                raise ValueError(
                    f"item_total {self.item_total} does not match "
                    f"quantity {self.quantity} × unit_price {self.unit_price} = {expected}"
                )
        return self


class Receipt(BaseModel):
    """Full structured receipt extracted from an image."""

    merchant: Optional[str] = Field(default=None, description="Merchant / store name.")
    # Stored as ISO date string (YYYY-MM-DD) to avoid Pydantic/Python 3.14 compatibility issues.
    # Use receipt.date_parsed for a datetime.date object.
    date: Optional[str] = Field(default=None, description="Transaction date as YYYY-MM-DD string.")
    items: List[ReceiptItem] = Field(description="List of purchased line items.")
    subtotal: Optional[Decimal] = Field(
        default=None, ge=Decimal("0"), description="Sum of item totals before tax/discount."
    )
    tax: Optional[Decimal] = Field(
        default=None, ge=Decimal("0"), description="Tax amount."
    )
    discount: Optional[Decimal] = Field(
        default=None, ge=Decimal("0"), description="Discount / coupon amount."
    )
    total: Decimal = Field(ge=Decimal("0"), description="Final amount paid.")

    # Pipeline metadata — set by the pipeline, not the LLM
    status: ProcessingStatus = Field(default=ProcessingStatus.NEEDS_REVIEW)
    validation_errors: List[str] = Field(default_factory=list)

    @field_validator("date", mode="before")
    @classmethod
    def normalize_date(cls, v):
        """Accept date objects or ISO strings; store as YYYY-MM-DD string."""
        if v is None:
            return None
        if isinstance(v, DateType):
            return v.isoformat()
        if isinstance(v, str):
            # Validate format
            try:
                DateType.fromisoformat(v)
                return v
            except ValueError:
                raise ValueError(f"Invalid date format {v!r}. Expected YYYY-MM-DD.")
        raise ValueError(f"Cannot convert {v!r} to a date string.")

    @field_validator("subtotal", "tax", "discount", "total", mode="before")
    @classmethod
    def coerce_to_decimal(cls, v):
        if v is None:
            return v
        try:
            return Decimal(str(v))
        except Exception:
            raise ValueError(f"Cannot convert {v!r} to Decimal")

    @field_validator("items")
    @classmethod
    def require_at_least_one_item(cls, v):
        if not v:
            raise ValueError("Receipt must contain at least one item.")
        return v

    @property
    def date_parsed(self) -> Optional[DateType]:
        """Return the date as a datetime.date object, or None if not set."""
        if self.date is None:
            return None
        return DateType.fromisoformat(self.date)
