"""LLM extraction — Stage 3 of the pipeline.

Uses LangChain to:
  1. Build a structured prompt from OCR text.
  2. Invoke the local Llama 3 model via Ollama.
  3. Parse the JSON output into a Receipt Pydantic model.

Reliability features:
  - Exponential-backoff retry for transient failures (timeout, 5xx, rate-limit).
  - One repair attempt: if parsing fails, a follow-up prompt explains the error
    and asks the model to correct its output.
  - After MAX_EXTRACTION_ATTEMPTS the function returns None so the pipeline
    can mark the receipt as NEEDS_REVIEW rather than crashing.
"""

import json
import logging
import re
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import ValidationError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from invoicer import config
from invoicer.models import Receipt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangChain setup
# ---------------------------------------------------------------------------

def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def _build_extraction_chain(llm: ChatOllama):
    """Build the main extraction chain: prompt | llm | pydantic parser."""
    parser = PydanticOutputParser(pydantic_object=Receipt)

    prompt = PromptTemplate(
        template="""You are an expert at extracting structured data from receipt text.

Given the following receipt text, extract the following fields:
- merchant: store/restaurant name (string or null)
- date: transaction date in YYYY-MM-DD format (string or null)
- items: list of line items, each with:
    - item_name: string
    - quantity: integer (must be > 0)
    - unit_price: decimal string (must be >= 0, e.g. "12.99")
    - item_total: decimal string or null
- subtotal: decimal string or null
- tax: decimal string or null
- discount: decimal string or null
- total: decimal string (required, must be >= 0)

Rules:
- Use null for any field that is not present on the receipt.
- All monetary values must be decimal strings, not floats.
- Do not invent values not present in the receipt text.
- Output ONLY the JSON object, with no explanation or markdown code fences.

{format_instructions}

Receipt Text:
{receipt_text}
""",
        input_variables=["receipt_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return prompt | llm | parser, parser


def _build_repair_chain(llm: ChatOllama):
    """Build a repair chain that asks the LLM to fix a previous malformed output."""
    prompt = PromptTemplate(
        template="""Your previous extraction attempt produced output that failed validation.

Validation error:
{validation_error}

Your previous output:
{previous_output}

Original receipt text:
{receipt_text}

Please correct the JSON and output ONLY a valid JSON object matching this schema:
{format_instructions}

Output ONLY the corrected JSON, no explanation.
""",
        input_variables=["validation_error", "previous_output", "receipt_text"],
        partial_variables={
            "format_instructions": PydanticOutputParser(
                pydantic_object=Receipt
            ).get_format_instructions()
        },
    )
    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Transient-failure detection
# ---------------------------------------------------------------------------

_TRANSIENT_SUBSTRINGS = (
    "timeout",
    "timed out",
    "connection",
    "429",
    "500",
    "502",
    "503",
    "504",
)


def _is_transient(exc: Exception) -> bool:
    """Return True if the exception looks like a recoverable transient failure."""
    msg = str(exc).lower()
    return any(s in msg for s in _TRANSIENT_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Extraction with retry
# ---------------------------------------------------------------------------

def _extract_with_retry(chain, receipt_text: str, llm: ChatOllama) -> Receipt:
    """Run the extraction chain, retrying up to 3× on transient failures."""

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(config.MAX_EXTRACTION_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _invoke():
        return chain.invoke({"receipt_text": receipt_text})

    try:
        return _invoke()
    except RetryError as e:
        raise RuntimeError(f"Extraction failed after retries: {e.last_attempt.exception()}") from e


def _parse_json_from_raw(raw: str, receipt_text: str) -> Optional[Receipt]:
    """Try to extract a JSON object from raw LLM string output."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Find the outermost JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())
        return Receipt(**data)
    except (json.JSONDecodeError, ValidationError):
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_receipt(ocr_text: str) -> Optional[Receipt]:
    """Extract a structured Receipt from raw OCR text.

    Strategy:
      1. Try the main extraction chain (with backoff retries on transient errors).
      2. If Pydantic validation fails, attempt one repair via a follow-up prompt.
      3. If still invalid after MAX_EXTRACTION_ATTEMPTS total, return None.

    The pipeline marks the receipt NEEDS_REVIEW when this returns None.

    Args:
        ocr_text: Raw text from the OCR stage.

    Returns:
        Validated Receipt instance, or None if extraction ultimately fails.
    """
    llm = _build_llm()
    chain, _parser = _build_extraction_chain(llm)
    repair_chain = _build_repair_chain(llm)

    raw_output: str = ""

    for attempt in range(1, config.MAX_EXTRACTION_ATTEMPTS + 1):
        logger.info("Extraction attempt %d/%d", attempt, config.MAX_EXTRACTION_ATTEMPTS)

        try:
            if attempt == 1:
                # First attempt: full extraction chain
                result = _extract_with_retry(chain, ocr_text, llm)
            else:
                # Subsequent attempts: repair prompt with previous failure context
                logger.info("Attempting repair with validation error context.")
                raw_output = repair_chain.invoke(
                    {
                        "validation_error": last_error,
                        "previous_output": raw_output,
                        "receipt_text": ocr_text,
                    }
                )
                result = _parse_json_from_raw(raw_output, ocr_text)
                if result is None:
                    last_error = "Could not parse JSON from repair output."
                    logger.warning("Repair attempt %d failed to parse JSON.", attempt)
                    continue

            logger.info("Extraction succeeded on attempt %d.", attempt)
            return result

        except ValidationError as e:
            last_error = str(e)
            raw_output = str(e)  # best we have without capturing raw text
            logger.warning("Pydantic validation failed on attempt %d: %s", attempt, e)

        except Exception as e:
            if _is_transient(e) and attempt < config.MAX_EXTRACTION_ATTEMPTS:
                last_error = str(e)
                logger.warning("Transient error on attempt %d: %s. Will retry.", attempt, e)
            else:
                logger.error("Non-retryable extraction error: %s", e)
                return None

    logger.error(
        "Extraction failed after %d attempts. Last error: %s",
        config.MAX_EXTRACTION_ATTEMPTS,
        last_error if "last_error" in dir() else "unknown",
    )
    return None
