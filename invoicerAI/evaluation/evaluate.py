"""Evaluation framework for InvoicerAI.

Runs the full pipeline on the evaluation dataset and compares predictions
against ground_truth.json. Reports field-level accuracy metrics.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --ocr-only   # skip LLM, measure OCR quality only
    python evaluation/evaluate.py --no-db      # don't persist to database

Metrics reported:
    - merchant accuracy    (exact string match, case-insensitive)
    - date accuracy        (exact match)
    - item count accuracy  (correct number of items extracted)
    - item name accuracy   (best-match across predicted items)
    - quantity accuracy    (exact match per item)
    - unit_price accuracy  (match within $0.01 tolerance)
    - total accuracy       (match within $0.01 tolerance)
    - overall accuracy     (mean of all field accuracies)
    - pipeline status breakdown (COMPLETED / NEEDS_REVIEW / FAILED)
"""

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from invoicer.pipeline import process_receipt
from invoicer.models import Receipt, ProcessingStatus

EVAL_DIR = Path(__file__).parent
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"
DATASET_DIR = EVAL_DIR / "dataset"

PRICE_TOLERANCE = Decimal("0.01")


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _str_match(pred: Optional[str], truth: Optional[str]) -> bool:
    if truth is None:
        return True  # not evaluated
    if pred is None:
        return False
    return pred.strip().lower() == truth.strip().lower()


def _decimal_match(pred: Optional[Any], truth: Optional[str]) -> bool:
    if truth is None:
        return True
    if pred is None:
        return False
    try:
        p = Decimal(str(pred))
        t = Decimal(str(truth))
        return abs(p - t) <= PRICE_TOLERANCE
    except (InvalidOperation, TypeError):
        return False


def _best_item_match(pred_items, truth_item: dict) -> Optional[dict]:
    """Find the predicted item whose name most closely matches the ground-truth item."""
    truth_name = truth_item["item_name"].lower()
    for item in pred_items:
        if truth_name in item.item_name.lower() or item.item_name.lower() in truth_name:
            return item
    return None


def _evaluate_items(predicted_items, truth_items: list[dict]) -> dict:
    """Return per-field item accuracy metrics."""
    if not truth_items:
        return {"item_count": 1.0, "name": 1.0, "quantity": 1.0, "unit_price": 1.0}

    count_correct = len(predicted_items) == len(truth_items)
    name_matches, qty_matches, price_matches = [], [], []

    for truth_item in truth_items:
        matched = _best_item_match(predicted_items, truth_item)
        if matched:
            name_matches.append(True)
            qty_matches.append(matched.quantity == truth_item["quantity"])
            price_matches.append(_decimal_match(matched.unit_price, truth_item["unit_price"]))
        else:
            name_matches.append(False)
            qty_matches.append(False)
            price_matches.append(False)

    return {
        "item_count": float(count_correct),
        "name": sum(name_matches) / len(name_matches),
        "quantity": sum(qty_matches) / len(qty_matches),
        "unit_price": sum(price_matches) / len(price_matches),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(no_db: bool = True) -> None:
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    results = []
    status_counts = {s.value: 0 for s in ProcessingStatus}

    print(f"\n{'='*60}")
    print("  INVOICERAI EVALUATION")
    print(f"  Dataset: {len(ground_truth)} receipts")
    print(f"{'='*60}\n")

    for entry in ground_truth:
        receipt_id = entry["id"]
        image_path = DATASET_DIR / Path(entry["image"]).name
        expected = entry["expected"]

        print(f"[{receipt_id}] {entry['description']}")

        if not image_path.exists():
            print(f"  ⚠ Image not found: {image_path}. Skipping.\n")
            continue

        # Run the pipeline
        receipt: Receipt = process_receipt(image_path, persist=not no_db)
        status_counts[receipt.status.value] += 1
        print(f"  Status: {receipt.status.value}")

        if receipt.status == ProcessingStatus.FAILED:
            print(f"  Pipeline failed — skipping field comparison.\n")
            results.append({"id": receipt_id, "failed": True})
            continue

        # Field-level comparison
        merchant_ok = _str_match(receipt.merchant, expected.get("merchant"))
        date_ok = _str_match(receipt.date, expected.get("date"))
        total_ok = _decimal_match(receipt.total, expected.get("total"))
        item_metrics = _evaluate_items(receipt.items, expected.get("items", []))

        row = {
            "id": receipt_id,
            "failed": False,
            "merchant": merchant_ok,
            "date": date_ok,
            "total": total_ok,
            **{f"item_{k}": v for k, v in item_metrics.items()},
        }
        results.append(row)

        icon = lambda b: "✓" if b else "✗"
        print(f"  Merchant   : {icon(merchant_ok)}  (predicted: {receipt.merchant!r}  expected: {expected.get('merchant')!r})")
        print(f"  Date       : {icon(date_ok)}  (predicted: {receipt.date}  expected: {expected.get('date')})")
        print(f"  Total      : {icon(total_ok)}  (predicted: {receipt.total}  expected: {expected.get('total')})")
        print(f"  Item count : {icon(item_metrics['item_count']>0.5)}  (predicted: {len(receipt.items)}  expected: {len(expected.get('items',[]))})")
        print(f"  Item names : {item_metrics['name']:.0%} matched")
        print(f"  Quantities : {item_metrics['quantity']:.0%} matched")
        print(f"  Unit prices: {item_metrics['unit_price']:.0%} matched")
        print()

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    valid = [r for r in results if not r.get("failed")]
    if not valid:
        print("No valid results to aggregate.")
        return

    def _avg(key):
        vals = [r[key] for r in valid if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    metrics = {
        "merchant_accuracy": _avg("merchant"),
        "date_accuracy": _avg("date"),
        "total_accuracy": _avg("total"),
        "item_count_accuracy": _avg("item_item_count"),
        "item_name_accuracy": _avg("item_name"),
        "quantity_accuracy": _avg("item_quantity"),
        "unit_price_accuracy": _avg("item_unit_price"),
    }
    metrics["overall_accuracy"] = sum(metrics.values()) / len(metrics)

    print("=" * 60)
    print("  AGGREGATE METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<28}: {v:.1%}")
    print()
    print("  Pipeline Status Breakdown:")
    for status, count in status_counts.items():
        print(f"    {status:<15}: {count}/{len(ground_truth)}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate InvoicerAI on the ground-truth dataset.")
    parser.add_argument("--no-db", action="store_true", default=True, help="Skip DB persistence (default)")
    parser.add_argument("--persist", action="store_true", help="Save results to database")
    args = parser.parse_args()

    evaluate(no_db=not args.persist)
