"""
NovaMart RAG Query Function — v2
Adds question classification: broad/comparative questions get
structured retrieval (most recent week per store), narrow questions
keep the original semantic vector search.

This fixes the bug where broad questions like "which stores are
underperforming" only retrieved 8 of 1000+ chunks via pure similarity,
silently dropping relevant stores/weeks from the answer.
"""

import os
import re
import chromadb
import anthropic
from openai import OpenAI
from dotenv import load_dotenv

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
openai_client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
# CONNECT TO CHROMADB
# ─────────────────────────────────────────────
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chromadb")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_collection("novamart")

ALL_STORE_IDS = [f"S{i:03d}" for i in range(1, 26)]  # S001..S025


# ─────────────────────────────────────────────
# STEP 0 — CLASSIFY THE QUESTION
# ─────────────────────────────────────────────
BROAD_PATTERNS = [
    r"\bwhich stores?\b",
    r"\bwhat stores?\b",
    r"\bwhere (are|is)\b",
    r"\bacross (all )?(stores?|locations?)\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\beach store\b",
    r"\bevery store\b",
    r"\ball (\d+ )?stores?\b",
    r"\boverall\b",
    r"\branking?\b",
    r"\bworst\b",
    r"\bbest\b",
    r"\btop \d+\b",
    r"\bbottom \d+\b",
    r"\bunderperform",
    r"\boutperform",
    r"\blist (the )?stores?\b",
    r"\bhow (do|does|are) (the )?stores?\b",
]


def is_broad_question(question: str) -> bool:
    """
    Returns True if the question is comparative/broad (spans many stores)
    rather than narrow (about one specific store/week).
    Keyword-based — fast, no extra API call, easy to extend.
    """
    q = question.lower()
    return any(re.search(pattern, q) for pattern in BROAD_PATTERNS)


DATA_TYPE_KEYWORDS = {
    "labor":    [r"\blabor\b", r"\bovertime\b", r"\bstaffing\b", r"\bheadcount\b",
                 r"\bhours (scheduled|worked)\b", r"\bturnover\b"],
    "inventory": [r"\binventory\b", r"\bstock\b", r"\bshrinkage\b", r"\bstockout\b",
                  r"\bout of stock\b", r"\bsku\b"],
    "customer": [r"\bcustomer\b", r"\bnps\b", r"\bsatisfaction\b", r"\bcomplaint\b",
                 r"\breview\b", r"\bcsat\b"],
    "sales":    [r"\bsales\b", r"\brevenue\b", r"\btarget\b", r"\bconversion\b",
                 r"\bfoot traffic\b", r"\btransactions?\b"],
}


def detect_data_type(question: str) -> str:
    """
    Determines which dataset a broad question is actually about,
    so structured retrieval pulls the right chunks instead of
    defaulting to sales regardless of topic.
    Falls back to "sales" only if nothing else matches, since that's
    the most common question type — but checks every type first.
    """
    q = question.lower()
    for data_type, patterns in DATA_TYPE_KEYWORDS.items():
        if any(re.search(p, q) for p in patterns):
            return data_type
    return "sales"


# ─────────────────────────────────────────────
# STEP 1 — EMBED THE QUESTION (used for narrow path)
# ─────────────────────────────────────────────
def embed_question(question: str) -> list:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────
# STEP 2a — NARROW RETRIEVAL (original semantic search)
# ─────────────────────────────────────────────
def retrieve_chunks_semantic(question_vector: list, n_results: int = 8) -> list:
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=n_results,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    chunks = []
    for doc_text, metadata in zip(documents, metadatas):
        header = f"[{metadata.get('type', '').upper()} | {metadata.get('city', '')} | {metadata.get('week', '')}]"
        chunks.append(f"{header}\n{doc_text}")

    return chunks


# ─────────────────────────────────────────────
# STEP 2b — BROAD RETRIEVAL (structured, one chunk per store)
# ─────────────────────────────────────────────
def retrieve_chunks_structured(data_type: str = "sales") -> list:
    """
    Pulls the most recent week's chunk for EVERY store, guaranteeing
    full coverage instead of relying on embedding similarity to decide
    which stores happen to surface. Used for comparative/broad questions.
    """
    chunks = []

    for store_id in ALL_STORE_IDS:
        # Pull all chunks of this type for this store, sorted by week,
        # take the most recent one. ChromaDB's `get` with a where filter
        # lets us fetch by metadata instead of similarity.
        results = collection.get(
            where={
                "$and": [
                    {"type": {"$eq": data_type}},
                    {"store_id": {"$eq": store_id}},
                ]
            }
        )

        if not results["documents"]:
            continue  # no data for this store/type — skip rather than guess

        # Sort by week string (works for "2024-W01".."2024-W12" lexically)
        paired = list(zip(results["documents"], results["metadatas"]))
        paired.sort(key=lambda pair: pair[1].get("week", ""))
        latest_doc, latest_meta = paired[-1]

        header = f"[{latest_meta.get('type', '').upper()} | {latest_meta.get('city', '')} | {latest_meta.get('week', '')}]"
        chunks.append(f"{header}\n{latest_doc}")

    return chunks


# ─────────────────────────────────────────────
# STEP 2 — UNIFIED RETRIEVAL ENTRY POINT
# ─────────────────────────────────────────────
def retrieve_chunks(question: str, question_vector: list, n_results: int = 8) -> list:
    if is_broad_question(question):
        data_type = detect_data_type(question)
        # Structured pull: guarantees every store is represented,
        # for whichever dataset the question is actually about
        chunks = retrieve_chunks_structured(data_type=data_type)
        # Still add a handful of semantically relevant chunks on top,
        # in case the question also references something specific
        chunks += retrieve_chunks_semantic(question_vector, n_results=4)
        return chunks
    else:
        return retrieve_chunks_semantic(question_vector, n_results=n_results)


# ─────────────────────────────────────────────
# STEP 3 — BUILD PROMPT AND CALL CLAUDE
# ─────────────────────────────────────────────
def ask_claude(question: str, chunks: list) -> str:
    context = "\n\n".join(chunks)

    system_prompt = """You are NovaMart's AI operations analyst. 
You have access to weekly operational data across all 25 NovaMart store locations including 
sales performance, inventory metrics, labor costs, and customer satisfaction scores.

Your job is to answer questions from NovaMart leadership with clear, specific, data-backed insights.

Rules:
- Always reference specific store names, IDs, and numbers from the data provided
- Flag underperformance clearly and suggest what to investigate
- If data is insufficient to answer fully, say so honestly — do not assume missing data doesn't exist
- Be concise but specific — executives want actionable insights, not summaries
- Always mention the time period the data covers"""

    user_message = f"""Here is the relevant NovaMart operational data:

{context}

---

Question: {question}"""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    return response.content[0].text


# ─────────────────────────────────────────────
# MAIN QUERY FUNCTION
# ─────────────────────────────────────────────
def query(question: str, verbose: bool = False) -> str:
    if verbose:
        print(f"\n🔍 Question: {question}")
        broad = is_broad_question(question)
        if broad:
            dtype = detect_data_type(question)
            print(f"   Classified as: BROAD (structured, data_type={dtype})")
        else:
            print(f"   Classified as: NARROW (semantic)")
        print("   Embedding question...")

    question_vector = embed_question(question)

    if verbose:
        print("   Retrieving chunks...")

    chunks = retrieve_chunks(question, question_vector)

    if verbose:
        print(f"   Retrieved {len(chunks)} chunks")
        print("   Sending to Claude...\n")

    answer = ask_claude(question, chunks)
    return answer


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "Which stores are underperforming against revenue targets?",
        "Which stores have the worst shrinkage rates?",
        "Where are labor costs running over budget?",
        "Which stores have the lowest NPS scores?",
        "How did store S016 do last week?",
    ]

    print("\n🏪 NovaMart RAG — Quick Test")
    print("─" * 50)

    for question in test_questions:
        print(f"\n❓ {question}")
        print("─" * 50)
        answer = query(question, verbose=True)
        print(answer)
        print()


# ─────────────────────────────────────────────
# QUERY WITH CONVERSATION HISTORY
# Used by the Slack bot for context-aware responses
# ─────────────────────────────────────────────
def query_with_history(question: str, history: list) -> str:
    """
    RAG query that includes conversation history so Claude
    can answer follow-up questions with full context.

    history format: [{"role": "user/assistant", "content": "..."}]
    """
    question_vector = embed_question(question)
    chunks = retrieve_chunks(question, question_vector)
    context = "\n\n".join(chunks)

    system_prompt = """You are NovaMart's AI operations analyst inside Slack.
You have access to weekly operational data across all 25 NovaMart store locations including
sales performance, inventory metrics, labor costs, and customer satisfaction scores.

Your job is to answer questions from NovaMart leadership with clear, specific, data-backed insights.

Rules:
- Always reference specific store names, IDs, and numbers from the data provided
- Flag underperformance clearly and suggest what to investigate
- If data is insufficient to answer fully, say so honestly — do not assume missing data doesn't exist
- Be concise but specific — executives want actionable insights
- Remember context from earlier in the conversation for follow-up questions
- Keep responses readable in Slack — use short paragraphs, avoid excessive markdown"""

    messages = []

    for turn in history:
        messages.append({
            "role": turn["role"],
            "content": turn["content"]
        })

    messages.append({
        "role": "user",
        "content": f"""Here is relevant NovaMart operational data:

{context}

---

Question: {question}"""
    })

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )

    return response.content[0].text