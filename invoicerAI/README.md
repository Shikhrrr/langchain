# InvoicerAI — Receipt Intelligence Pipeline

A modular Python pipeline that converts receipt images into structured, validated JSON data and stores the results in PostgreSQL. Built with LangChain and a locally-running Llama 3 model.

---

## Problem

Receipts come in wildly different formats — different merchants, layouts, fonts, and column arrangements. Traditional OCR alone produces unstructured text. Without a normalization layer, you cannot reliably extract `merchant`, `date`, `items`, `unit prices`, or `totals` from that raw text into a consistent schema.

InvoicerAI solves this with a four-stage pipeline:

```
Receipt Image
      │
      ▼  Stage 1
┌─────────────┐
│ Preprocess  │  Grayscale, contrast, sharpness, noise reduction (Pillow)
└─────────────┘
      │
      ▼  Stage 2
┌─────────────┐
│     OCR     │  OCR.space (primary) → pytesseract (fallback)
└─────────────┘
      │  raw text
      ▼  Stage 3
┌─────────────┐
│   LLM       │  LangChain + local Llama 3 via Ollama
│ Extraction  │  Structured prompt → PydanticOutputParser → Receipt schema
└─────────────┘
      │  Receipt object
      ▼  Stage 4a
┌─────────────┐
│  Pydantic   │  Structural validation: types, required fields, ranges
│ Validation  │
└─────────────┘
      │
      ▼  Stage 4b
┌─────────────┐
│  Business   │  Cross-check: item totals ≈ subtotal, subtotal + tax - discount ≈ total
│ Validation  │
└─────────────┘
      │
      ▼
┌─────────────┐
│  PostgreSQL │  receipts + receipt_items (NUMERIC(10,2) monetary columns)
└─────────────┘
      │
      ▼
  Receipt with status: COMPLETED | NEEDS_REVIEW | FAILED
```

---

## Architecture Details

### Why an LLM?

OCR converts an image into unstructured text like:

```
FRESH MART
Whole Milk 1L  2  $2.99  $5.98
Tax 8%               $1.16
TOTAL                $15.62
```

The structure varies per merchant. An LLM with a structured prompt can:
- Identify which token is the merchant name vs. an item description
- Map "2 × $2.99" onto `quantity=2, unit_price=2.99`
- Distinguish tax from discount from subtotal
- Handle missing, reordered, or unlabelled fields

Rule-based parsing would require per-merchant templates. The LLM generalises.

### Why LangChain?

LangChain provides:
- **`PromptTemplate`** — parameterised prompt construction with format instructions injected automatically
- **`ChatOllama`** — model invocation abstracted from the transport layer
- **`PydanticOutputParser`** — generates format instructions from a Pydantic schema and parses the LLM's JSON response directly into a typed Python object
- **`create_sql_agent`** — SQL query generation for the factual query path

### Validation: Two Separate Layers

| Layer | Where | What it checks |
|---|---|---|
| **Structural** | `models.py` (Pydantic) | Types, required fields, `quantity > 0`, `unit_price ≥ 0`, `total ≥ 0` |
| **Business** | `validation.py` | `sum(item_totals) ≈ subtotal`, `subtotal + tax − discount ≈ total` |

Pydantic rejects structurally invalid data before it reaches business validation. Business validation catches semantically wrong data (e.g. items that add up to $50 but the receipt says $30).

---

## Reliability

### OCR Fallback

```
OCR.space (REST API, high accuracy)
    │ on failure
    ▼
pytesseract (local Tesseract, no internet required)
    │ on failure
    ▼
FAILED status — pipeline stops
```

### LLM Retry + Repair

```
LLM extraction attempt 1
    │ transient error (timeout, 429, 5xx)
    ▼  exponential backoff (tenacity)
LLM extraction attempt 2
    │ Pydantic validation fails
    ▼  repair prompt: "Your output was invalid because X. Correct it."
LLM extraction attempt 3
    │ still invalid
    ▼
NEEDS_REVIEW — persisted with error details
```

Maximum attempts controlled by `MAX_EXTRACTION_ATTEMPTS` (default: 3). Retries only happen for transient failures; invalid inputs fail fast.

### Processing Status

| Status | Meaning |
|---|---|
| `COMPLETED` | All stages passed, no business-rule violations |
| `NEEDS_REVIEW` | Extraction succeeded but business rules flagged inconsistencies, OR extraction failed after retries |
| `FAILED` | OCR failed completely — no text could be extracted |

---

## Evaluation

The evaluation framework runs the full pipeline on 5 synthetic receipts covering:
- Simple grocery (3 items, tax)
- Coffee shop (discount)
- Restaurant (4 items)
- Pharmacy (single item, zero tax)
- Electronics (high-value, large discount)

**Metrics reported:**
- Merchant accuracy (case-insensitive exact match)
- Date accuracy
- Item count accuracy
- Item name accuracy (best-match)
- Quantity accuracy
- Unit price accuracy (±$0.01 tolerance)
- Total accuracy (±$0.01 tolerance)
- Overall accuracy (mean)
- Pipeline status breakdown

Run:
```bash
python evaluation/evaluate.py
```

---

## Project Structure

```
InvoicerAI/
├── src/invoicer/
│   ├── config.py          # Environment variable loading
│   ├── models.py          # Pydantic schemas (Receipt, ReceiptItem, ProcessingStatus)
│   ├── preprocessing.py   # Stage 1: image preprocessing (Pillow)
│   ├── ocr.py             # Stage 2: OCR abstraction + fallback
│   ├── extraction.py      # Stage 3: LangChain chain + retry/repair
│   ├── validation.py      # Stage 4b: business-rule validation
│   ├── database.py        # SQLAlchemy ORM + persistence
│   ├── pipeline.py        # Orchestrator
│   └── query.py           # NL query handler (SQL agent + reasoning chain)
├── tests/
│   ├── test_validation.py
│   ├── test_extraction.py
│   └── test_pipeline.py
├── evaluation/
│   ├── dataset/           # Synthetic receipt images
│   ├── ground_truth.json
│   └── evaluate.py
├── examples/
│   └── sample_run.py
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running Locally

### 1. Prerequisites

| Requirement | Install |
|---|---|
| Python 3.10+ | https://python.org |
| Ollama | https://ollama.com |
| Llama 3 model | `ollama pull llama3` |
| PostgreSQL | https://postgresql.org |
| Tesseract (OCR fallback) | `brew install tesseract` |

### 2. Setup

```bash
# Clone / enter the project
cd InvoicerAI

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env
# Edit .env: add your OCR_SPACE_API_KEY and DATABASE_URL

# Create the PostgreSQL database
createdb receipt_db
```

### 3. Run the pipeline

```bash
# Process a receipt image
python main.py process path/to/receipt.jpg

# Process without saving to DB
python main.py process path/to/receipt.jpg --no-db

# Also output raw JSON
python main.py process path/to/receipt.jpg --json

# Interactive query mode
python main.py --query
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

### 5. Run evaluation

```bash
python evaluation/evaluate.py
```

---

## Environment Variables

See [`.env.example`](.env.example) for all configurable values.

| Variable | Description | Default |
|---|---|---|
| `OCR_SPACE_API_KEY` | OCR.space API key | *(required for primary OCR)* |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name | `llama3` |
| `DATABASE_URL` | PostgreSQL connection string | *(required for persistence)* |
| `MAX_EXTRACTION_ATTEMPTS` | Max LLM retries | `3` |
| `MONETARY_TOLERANCE` | Business-rule tolerance in $ | `0.05` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM framework | LangChain |
| Local LLM | Llama 3 via Ollama |
| Structured validation | Pydantic v2 |
| Database | PostgreSQL + SQLAlchemy |
| Image processing | Pillow |
| Primary OCR | OCR.space REST API |
| Fallback OCR | pytesseract (Tesseract) |
| Retry logic | tenacity |
| Configuration | python-dotenv |
| Testing | pytest + pytest-mock |
