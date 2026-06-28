"""
NovaMart RAG Ingestion Pipeline
Converts CSV data into text chunks, embeds them, and stores in ChromaDB.

Why text chunks instead of raw CSV rows?
- LLMs understand natural language, not raw numbers
- "S001 had revenue of $117,794 vs target of $115,500 (2.0% above target)"
  is far more useful to Claude than: S001,2024-W01,117794,115500,2.0
"""

import os
import pandas as pd
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB stored locally in /data/chromadb
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chromadb")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ─────────────────────────────────────────────
# EMBEDDING FUNCTION
# ─────────────────────────────────────────────
def embed(texts: list[str]) -> list[list[float]]:
    """
    Calls OpenAI text-embedding-3-small to convert text into vectors.
    We batch in groups of 100 to stay within API limits.
    """
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend([r.embedding for r in response.data])
        print(f"   Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")

    return all_embeddings


# ─────────────────────────────────────────────
# CHUNKING FUNCTIONS
# Each function converts one CSV row into a
# readable natural language text chunk.
# ─────────────────────────────────────────────

def chunk_store_master(df_stores: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """One chunk per store — static profile info."""
    chunks, ids, metadatas = [], [], []

    for _, row in df_stores.iterrows():
        text = (
            f"NovaMart {row['city']} (Store ID: {row['store_id']}) is located in "
            f"{row['city']}, {row['state']} in the {row['region']} region. "
            f"The store is {row['square_footage']:,} square feet and opened on {row['opened_date']}. "
            f"The store manager is {row['store_manager']}."
        )
        chunks.append(text)
        ids.append(f"store_master_{row['store_id']}")
        metadatas.append({
            "type": "store_master",
            "store_id": row["store_id"],
            "city": row["city"],
            "region": row["region"],
        })

    return chunks, ids, metadatas


def chunk_sales(df_sales: pd.DataFrame, df_stores: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """One chunk per store per week — sales performance."""
    chunks, ids, metadatas = [], [], []

    store_lookup = df_stores.set_index("store_id")["city"].to_dict()

    for _, row in df_sales.iterrows():
        city = store_lookup.get(row["store_id"], row["store_id"])
        variance_dir = "above" if row["revenue_variance_pct"] >= 0 else "below"
        growth_dir   = "up" if row["same_store_sales_growth"] >= 0 else "down"

        text = (
            f"NovaMart {city} ({row['store_id']}) sales for {row['week']} "
            f"(week starting {row['week_start_date']}): "
            f"Revenue was ${row['revenue_actual']:,.2f} against a target of ${row['revenue_target']:,.2f}, "
            f"{abs(row['revenue_variance_pct'])}% {variance_dir} target. "
            f"The store recorded {row['transactions']:,} transactions with an average basket size of ${row['avg_basket_size']}. "
            f"Foot traffic was {row['foot_traffic']:,} visitors with a conversion rate of {row['conversion_rate']}%. "
            f"Same-store sales growth was {growth_dir} {abs(row['same_store_sales_growth'])}% vs the prior period."
        )
        chunks.append(text)
        ids.append(f"sales_{row['store_id']}_{row['week']}")
        metadatas.append({
            "type": "sales",
            "store_id": row["store_id"],
            "week": row["week"],
            "city": city,
            "revenue_variance_pct": float(row["revenue_variance_pct"]),
        })

    return chunks, ids, metadatas


def chunk_inventory(df_inv: pd.DataFrame, df_stores: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """One chunk per store per week — inventory metrics."""
    chunks, ids, metadatas = [], [], []

    store_lookup = df_stores.set_index("store_id")["city"].to_dict()

    for _, row in df_inv.iterrows():
        city = store_lookup.get(row["store_id"], row["store_id"])

        text = (
            f"NovaMart {city} ({row['store_id']}) inventory for {row['week']}: "
            f"Inventory turnover was {row['inventory_turnover']} times. "
            f"Shrinkage rate was {row['shrinkage_rate']}% (industry benchmark is under 2%). "
            f"There were {row['stockout_incidents']} stockout incidents. "
            f"Overstock value sitting on shelves was ${row['overstock_value']:,.2f}. "
            f"Days of supply on hand: {row['days_supply_on_hand']} days. "
            f"Receiving accuracy was {row['receiving_accuracy']}%."
        )
        chunks.append(text)
        ids.append(f"inventory_{row['store_id']}_{row['week']}")
        metadatas.append({
            "type": "inventory",
            "store_id": row["store_id"],
            "week": row["week"],
            "city": city,
            "shrinkage_rate": float(row["shrinkage_rate"]),
        })

    return chunks, ids, metadatas


def chunk_labor(df_labor: pd.DataFrame, df_stores: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """One chunk per store per week — labor metrics."""
    chunks, ids, metadatas = [], [], []

    store_lookup = df_stores.set_index("store_id")["city"].to_dict()

    for _, row in df_labor.iterrows():
        city = store_lookup.get(row["store_id"], row["store_id"])
        variance_dir = "over" if row["labor_variance_pct"] >= 0 else "under"

        text = (
            f"NovaMart {city} ({row['store_id']}) labor data for {row['week']}: "
            f"Headcount was {row['headcount']} employees. "
            f"Hours scheduled: {row['hours_scheduled']}, hours actually worked: {row['hours_worked']}. "
            f"Labor cost was ${row['labor_cost_actual']:,.2f} against a target of ${row['labor_cost_target']:,.2f}, "
            f"{abs(row['labor_variance_pct'])}% {variance_dir} budget. "
            f"Sales per labor hour was ${row['sales_per_labor_hour']:,.2f}. "
            f"Staff turnover rate was {row['turnover_rate']}% this period."
        )
        chunks.append(text)
        ids.append(f"labor_{row['store_id']}_{row['week']}")
        metadatas.append({
            "type": "labor",
            "store_id": row["store_id"],
            "week": row["week"],
            "city": city,
            "labor_variance_pct": float(row["labor_variance_pct"]),
        })

    return chunks, ids, metadatas


def chunk_customer(df_cust: pd.DataFrame, df_stores: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """One chunk per store per week — customer metrics."""
    chunks, ids, metadatas = [], [], []

    store_lookup = df_stores.set_index("store_id")["city"].to_dict()

    for _, row in df_cust.iterrows():
        city = store_lookup.get(row["store_id"], row["store_id"])

        # NPS interpretation
        if row["nps_score"] >= 50:
            nps_label = "excellent"
        elif row["nps_score"] >= 30:
            nps_label = "good"
        elif row["nps_score"] >= 0:
            nps_label = "poor"
        else:
            nps_label = "very poor (negative)"

        text = (
            f"NovaMart {city} ({row['store_id']}) customer data for {row['week']}: "
            f"Net Promoter Score (NPS) was {row['nps_score']} ({nps_label}). "
            f"Customer return rate was {row['return_rate']}%. "
            f"Active loyalty members: {row['loyalty_members_active']:,}, "
            f"with {row['new_loyalty_signups']} new signups this week. "
            f"Complaints logged: {row['complaints_logged']}."
        )
        chunks.append(text)
        ids.append(f"customer_{row['store_id']}_{row['week']}")
        metadatas.append({
            "type": "customer",
            "store_id": row["store_id"],
            "week": row["week"],
            "city": city,
            "nps_score": int(row["nps_score"]),
        })

    return chunks, ids, metadatas


# ─────────────────────────────────────────────
# INGEST INTO CHROMADB
# ─────────────────────────────────────────────
def ingest_to_chroma(collection, chunks, ids, metadatas):
    """Embed chunks and upsert into ChromaDB collection."""
    print(f"   Embedding {len(chunks)} chunks...")
    embeddings = embed(chunks)

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            documents=chunks[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
    print(f"   ✅ {len(chunks)} chunks stored in ChromaDB")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n🏪 NovaMart RAG Ingestion Pipeline")
    print("─" * 40)

    # Load CSVs
    df_stores  = pd.read_csv(f"{DATA_DIR}/store_master.csv")
    df_sales   = pd.read_csv(f"{DATA_DIR}/weekly_sales.csv")
    df_inv     = pd.read_csv(f"{DATA_DIR}/inventory.csv")
    df_labor   = pd.read_csv(f"{DATA_DIR}/labor.csv")
    df_cust    = pd.read_csv(f"{DATA_DIR}/customer.csv")

    # Create or reset ChromaDB collection
    try:
        chroma_client.delete_collection("novamart")
        print("🗑️  Cleared existing ChromaDB collection")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="novamart",
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )
    print("📦 Created fresh ChromaDB collection: novamart\n")

    # Chunk + ingest each data type
    sections = [
        ("Store Master",  chunk_store_master(df_stores)),
        ("Sales Data",    chunk_sales(df_sales, df_stores)),
        ("Inventory",     chunk_inventory(df_inv, df_stores)),
        ("Labor",         chunk_labor(df_labor, df_stores)),
        ("Customer",      chunk_customer(df_cust, df_stores)),
    ]

    total_chunks = 0
    for name, (chunks, ids, metadatas) in sections:
        print(f"📄 Ingesting {name} ({len(chunks)} chunks)...")
        ingest_to_chroma(collection, chunks, ids, metadatas)
        total_chunks += len(chunks)
        print()

    print(f"✅ Ingestion complete — {total_chunks} total chunks in ChromaDB")
    print(f"   Collection: novamart")
    print(f"   Path: {CHROMA_PATH}")


if __name__ == "__main__":
    main()