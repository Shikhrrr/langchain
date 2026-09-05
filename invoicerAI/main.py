"""InvoicerAI — main entry point.

Usage:
  python main.py <receipt_image>          Process a receipt image
  python main.py <receipt_image> --no-db  Process without saving to DB
  python main.py --query                  Interactive query mode
  python main.py --query --db-url <url>   Query with custom DB URL
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Make 'src' importable without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from invoicer import config
from invoicer.models import ProcessingStatus


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_receipt(receipt) -> None:
    """Pretty-print a Receipt object to stdout."""
    print("\n" + "=" * 60)
    print("  INVOICERAI — EXTRACTION RESULT")
    print("=" * 60)
    print(f"  Status  : {receipt.status.value}")
    if receipt.merchant:
        print(f"  Merchant: {receipt.merchant}")
    if receipt.date:
        print(f"  Date    : {receipt.date}")
    print()
    print("  Items:")
    for item in receipt.items:
        total_str = f"  = ${item.item_total}" if item.item_total else ""
        print(f"    • {item.item_name:<30} qty={item.quantity:>3}  @${item.unit_price}{total_str}")
    print()
    if receipt.subtotal is not None:
        print(f"  Subtotal : ${receipt.subtotal}")
    if receipt.tax is not None:
        print(f"  Tax      : ${receipt.tax}")
    if receipt.discount is not None:
        print(f"  Discount : -${receipt.discount}")
    print(f"  Total    : ${receipt.total}")
    if receipt.validation_errors:
        print()
        print("  ⚠ Validation notes:")
        for err in receipt.validation_errors:
            print(f"    • {err}")
    print("=" * 60 + "\n")


def cmd_process(args) -> int:
    """Process a receipt image and print the result."""
    from invoicer.pipeline import process_receipt

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image file not found: {image_path}", file=sys.stderr)
        return 1

    persist = not args.no_db
    receipt = process_receipt(
        image_path,
        persist=persist,
        database_url=args.db_url or None,
    )

    _print_receipt(receipt)

    if args.json:
        data = receipt.model_dump(mode="json")
        print(json.dumps(data, indent=2, default=str))

    return 0 if receipt.status == ProcessingStatus.COMPLETED else 1


def cmd_query(args) -> int:
    """Enter interactive query mode."""
    from invoicer.query import run_interactive_query_loop

    run_interactive_query_loop(database_url=args.db_url or None)
    return 0


def main() -> int:
    _setup_logging()

    parser = argparse.ArgumentParser(
        prog="invoicer",
        description="InvoicerAI — Receipt Intelligence Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    # process subcommand
    process_parser = subparsers.add_parser("process", help="Process a receipt image")
    process_parser.add_argument("image", help="Path to receipt image")
    process_parser.add_argument("--no-db", action="store_true", help="Skip database persistence")
    process_parser.add_argument("--db-url", help="Override DATABASE_URL")
    process_parser.add_argument("--json", action="store_true", help="Also output JSON")

    # query subcommand
    query_parser = subparsers.add_parser("query", help="Interactive query mode")
    query_parser.add_argument("--db-url", help="Override DATABASE_URL")

    # Backwards-compatible shorthand: python main.py <image>
    parser.add_argument("image", nargs="?", help="Receipt image path (shorthand for 'process')")
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--db-url", help="Override DATABASE_URL")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--query", action="store_true", help="Enter query mode")

    args = parser.parse_args()

    if args.command == "process" or (args.image and not args.query):
        return cmd_process(args)
    elif args.command == "query" or args.query:
        return cmd_query(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
