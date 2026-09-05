"""Natural-language query handler for the receipt database.

Design (following the notebook's original approach):

1. Classify the query as Factual or Subjective using the LLM with
   structured output.
2. Factual queries → SQL agent performs exact DB computation; LLM only
   formats the answer. The database, not the LLM, computes sums/counts.
3. Subjective queries → fetch relevant data via ORM, then pass to a
   reasoning chain for consumer advice / recommendations.

Why this split matters (interview answer):
  SQL is deterministic and exact for numerical operations (totals, counts,
  averages). Delegating arithmetic to the LLM would be unreliable. The LLM
  adds value only where natural-language reasoning is genuinely needed.
"""

import logging
from typing import Optional

from langchain_classic.agents.agent_types import AgentType
from langchain_core.prompts import PromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sqlalchemy.orm import sessionmaker
from typing import Literal

from invoicer import config
from invoicer.database import ReceiptItemDB, get_engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query classification schema
# ---------------------------------------------------------------------------

class QueryClassification(BaseModel):
    query_type: Literal["Factual", "Subjective"] = Field(
        description=(
            "Factual: asks for objective data (counts, totals, prices, statistics). "
            "Subjective: asks for opinions, advice, or recommendations."
        )
    )


# ---------------------------------------------------------------------------
# Reasoning chain (subjective queries)
# ---------------------------------------------------------------------------

_REASONING_PROMPT = PromptTemplate(
    template="""You are an expert consumer advisor. Based on the purchase history below,
answer the user's question with specific, actionable advice.

Purchase History:
{purchase_data}

User Question:
{user_query}

Answer:""",
    input_variables=["purchase_data", "user_query"],
)


# ---------------------------------------------------------------------------
# Query classification helpers
# ---------------------------------------------------------------------------

_FACTUAL_KEYWORDS = frozenset([
    "how many", "total", "price", "quantity", "least", "most",
    "count", "sum", "average", "min", "max", "expensive", "cheap",
])


def _classify_query(user_query: str, classification_llm) -> str:
    """Classify a query as Factual or Subjective.

    Falls back to keyword heuristic if the LLM call fails.
    """
    try:
        result = classification_llm.invoke(
            f"Classify this user query:\n'{user_query}'\n\n"
            f"Factual: objective data, counts, totals, statistics.\n"
            f"Subjective: opinions, recommendations, advice."
        )
        return result.query_type
    except Exception as e:
        logger.warning("Classification LLM failed (%s); using keyword fallback.", e)
        lower = user_query.lower()
        return "Factual" if any(kw in lower for kw in _FACTUAL_KEYWORDS) else "Subjective"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def handle_query(user_query: str, database_url: Optional[str] = None) -> str:
    """Answer a natural-language question about receipts in the database.

    Args:
        user_query:   The user's natural-language question.
        database_url: Database connection string (defaults to config).

    Returns:
        Answer string to display to the user.
    """
    db_url = database_url or config.DATABASE_URL
    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)

    # Setup
    classification_llm = llm.with_structured_output(QueryClassification)
    reasoning_chain = _REASONING_PROMPT | llm | StrOutputParser()

    query_type = _classify_query(user_query, classification_llm)
    logger.info("Query classified as: %s", query_type)

    if query_type == "Factual":
        return _handle_factual(user_query, llm, db_url)
    else:
        return _handle_subjective(user_query, reasoning_chain, db_url)


def _handle_factual(user_query: str, llm, db_url: str) -> str:
    """Use a SQL agent to answer factual questions deterministically."""
    logger.info("Routing to SQL agent for factual query.")
    try:
        db = SQLDatabase.from_uri(db_url)
        sql_agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )
        result = sql_agent.invoke({"input": user_query})
        return result.get("output", "No answer returned.")
    except Exception as e:
        logger.error("SQL agent failed: %s", e)
        return f"Sorry, I couldn't retrieve that data: {e}"


def _handle_subjective(user_query: str, reasoning_chain, db_url: str) -> str:
    """Fetch purchase data and use the reasoning chain for advice/recommendations."""
    logger.info("Routing to reasoning chain for subjective query.")
    session = None
    try:
        engine = get_engine(db_url)
        SessionFactory = sessionmaker(bind=engine)
        session = SessionFactory()

        items = session.query(ReceiptItemDB).all()
        if not items:
            return "No receipt data found in the database."

        purchase_data = "\n".join(
            f"- {item.item_name}: qty {item.quantity}, unit price ${item.unit_price}"
            for item in items
        )

        return reasoning_chain.invoke(
            {"purchase_data": purchase_data, "user_query": user_query}
        )
    except Exception as e:
        logger.error("Reasoning chain failed: %s", e)
        return f"Sorry, I couldn't answer that: {e}"
    finally:
        if session:
            session.close()


def run_interactive_query_loop(database_url: Optional[str] = None) -> None:
    """Start an interactive query REPL. Type 'quit' or Ctrl+C to exit."""
    print("\n--- InvoicerAI Interactive Query Mode ---")
    print("Ask anything about your receipts. Type 'quit' to exit.\n")
    while True:
        try:
            user_query = input("Query> ").strip()
            if not user_query:
                continue
            if user_query.lower() in {"quit", "exit", "q"}:
                break
            answer = handle_query(user_query, database_url)
            print(f"\n{answer}\n")
        except KeyboardInterrupt:
            print("\nExiting query mode.")
            break
