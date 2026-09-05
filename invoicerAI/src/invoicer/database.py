"""Database persistence layer.

Schema:
  receipts       — one row per processed receipt image
  receipt_items  — one row per line item, FK → receipts

Design choices:
  - NUMERIC(10, 2) for all monetary columns — avoids floating-point drift.
  - Transactions: item rows are only committed if the receipt row succeeds.
  - The `raw_ocr_text` column is stored so you can debug extraction issues
    without re-running OCR on the original image.
  - ProcessingStatus is stored as a VARCHAR so the column is human-readable
    in a raw SQL client.

Usage:
  engine = get_engine()
  create_tables(engine)
  receipt_id = save_receipt(receipt, raw_ocr_text, engine)
"""

import logging
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    NUMERIC,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from invoicer import config
from invoicer.models import Receipt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class ReceiptDB(Base):
    __tablename__ = "receipts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    merchant = Column(String(255), nullable=True)
    date = Column(String(20), nullable=True)            # stored as ISO date string
    subtotal = Column(NUMERIC(10, 2), nullable=True)
    tax = Column(NUMERIC(10, 2), nullable=True)
    discount = Column(NUMERIC(10, 2), nullable=True)
    total = Column(NUMERIC(10, 2), nullable=False)
    status = Column(String(20), nullable=False, default="NEEDS_REVIEW")
    validation_errors = Column(Text, nullable=True)     # newline-separated list
    raw_ocr_text = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    items = relationship("ReceiptItemDB", back_populates="receipt", cascade="all, delete-orphan")


class ReceiptItemDB(Base):
    __tablename__ = "receipt_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id"), nullable=False)
    item_name = Column(String(255), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(NUMERIC(10, 2), nullable=False)
    item_total = Column(NUMERIC(10, 2), nullable=True)

    receipt = relationship("ReceiptDB", back_populates="items")


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------

def get_engine(database_url: Optional[str] = None):
    """Create and return a SQLAlchemy engine."""
    url = database_url or config.DATABASE_URL
    return create_engine(url)


def create_tables(engine) -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(engine)
    logger.info("Database tables verified/created.")


def get_session_factory(engine):
    return sessionmaker(bind=engine)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_receipt(
    receipt: Receipt,
    raw_ocr_text: str,
    engine,
) -> int:
    """Persist a Receipt (and its items) to the database in a single transaction.

    Args:
        receipt:      Validated Receipt Pydantic object.
        raw_ocr_text: The raw OCR string (for debugging).
        engine:       SQLAlchemy engine.

    Returns:
        The database ID of the inserted receipts row.

    Raises:
        Exception: Re-raises any DB error after rolling back.
    """
    SessionFactory = get_session_factory(engine)
    session: Session = SessionFactory()

    try:
        db_receipt = ReceiptDB(
            merchant=receipt.merchant,
            date=receipt.date,  # already stored as ISO date string
            subtotal=receipt.subtotal,
            tax=receipt.tax,
            discount=receipt.discount,
            total=receipt.total,
            status=receipt.status.value,
            validation_errors="\n".join(receipt.validation_errors) if receipt.validation_errors else None,
            raw_ocr_text=raw_ocr_text,
        )
        session.add(db_receipt)
        session.flush()  # get the auto-generated id before adding items

        for item in receipt.items:
            db_item = ReceiptItemDB(
                receipt_id=db_receipt.id,
                item_name=item.item_name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                item_total=item.item_total,
            )
            session.add(db_item)

        session.commit()
        logger.info("Receipt saved to DB with id=%d (status=%s)", db_receipt.id, receipt.status.value)
        return db_receipt.id

    except Exception:
        session.rollback()
        logger.exception("Failed to save receipt — transaction rolled back.")
        raise
    finally:
        session.close()


def get_all_receipts(engine) -> list[ReceiptDB]:
    """Return all receipt rows (for query/analysis)."""
    SessionFactory = get_session_factory(engine)
    session: Session = SessionFactory()
    try:
        return session.query(ReceiptDB).order_by(ReceiptDB.created_at.desc()).all()
    finally:
        session.close()
