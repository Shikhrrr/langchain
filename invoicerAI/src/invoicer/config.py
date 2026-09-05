"""Configuration — all settings loaded from environment variables.

Copy .env.example to .env and fill in your values.
Never hard-code credentials in source files.
"""

import os
from dotenv import load_dotenv

# Load .env file if present (safe no-op if absent)
load_dotenv()


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
OCR_SPACE_API_KEY: str = os.getenv("OCR_SPACE_API_KEY", "")
OCR_SPACE_URL: str = "https://api.ocr.space/parse/image"
OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")

# ---------------------------------------------------------------------------
# LLM (Ollama / local Llama 3)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://user:password@localhost/receipt_db"
)

# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------
# Maximum total LLM extraction attempts (1 initial + N-1 retries/repairs)
MAX_EXTRACTION_ATTEMPTS: int = int(os.getenv("MAX_EXTRACTION_ATTEMPTS", "3"))

# Monetary tolerance for business-rule checks (e.g. rounding on receipts)
MONETARY_TOLERANCE: float = float(os.getenv("MONETARY_TOLERANCE", "0.05"))

# OCR request timeout in seconds
OCR_TIMEOUT: int = int(os.getenv("OCR_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
