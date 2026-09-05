"""pytest configuration — add src to sys.path for all tests."""
import sys
from pathlib import Path

# Allow tests to import from src/invoicer without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))
