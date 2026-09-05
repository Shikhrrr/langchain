"""Example: process a receipt image end-to-end.

Run from the project root:
    python examples/sample_run.py examples/sample_receipt.png

Or with the venv:
    venv/bin/python examples/sample_run.py <path_to_image>
"""

import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/sample_run.py <receipt_image_path>")
        print()
        print("Example (using evaluation dataset):")
        print("  python examples/sample_run.py evaluation/dataset/receipt_001.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    from invoicer.pipeline import process_receipt

    print(f"\nProcessing: {image_path}")
    print("-" * 50)

    receipt = process_receipt(image_path, persist=False)

    print(f"\nStatus: {receipt.status.value}")
    print(f"Merchant: {receipt.merchant or 'unknown'}")
    print(f"Date: {receipt.date or 'unknown'}")
    print(f"\nItems ({len(receipt.items)}):")
    for item in receipt.items:
        print(f"  {item.item_name:<30}  qty={item.quantity}  @${item.unit_price}")

    print(f"\nSubtotal : {receipt.subtotal or 'N/A'}")
    print(f"Tax      : {receipt.tax or 'N/A'}")
    print(f"Discount : {receipt.discount or 'N/A'}")
    print(f"Total    : ${receipt.total}")

    if receipt.validation_errors:
        print(f"\nValidation Notes:")
        for e in receipt.validation_errors:
            print(f"  • {e}")

    print("\n--- JSON Output ---")
    print(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    main()
