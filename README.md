# 🏪 NovaMart RAG Pipeline

A RAG (Retrieval-Augmented Generation) pipeline that turns a retail chain's operational data into a Slack assistant you can ask anything — sales, inventory, labor, customer satisfaction — and get a precise, data-backed answer back in seconds.

Built as a personal project to understand how RAG retrieval actually breaks in practice, and how to validate it properly before trusting the output.

---

## 🧠 How It Works

| Stage | What Happens | Tools |
|---|---|---|
| 1. Generate | Synthetic operational data created for 25 fictional stores across 12 weeks | Python, NumPy, pandas |
| 2. Ingest | Each data row converted into a natural-language sentence, embedded, and stored | OpenAI Embeddings, ChromaDB |
| 3. Retrieve | Questions classified as narrow (single store) or broad (comparative) and routed to the right retrieval strategy | ChromaDB |
| 4. Generate | Retrieved context passed to Claude to produce the final answer | Claude (Anthropic) |
| 5. Interface | Questions asked and answered directly in Slack | Slack Bolt SDK |

---

## 🛠️ Tech Stack

- **[ChromaDB](https://www.trychroma.com)** — Vector store for embedded operational data
- **[OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)** — `text-embedding-3-small` for converting text to vectors
- **[Claude (Anthropic)](https://anthropic.com)** — LLM powering answer generation
- **[Slack Bolt SDK](https://slack.dev/bolt-python/)** — Slack bot interface with lazy listeners for non-blocking responses
- **Python** — Core language (pandas, NumPy for data generation)

---

## 📁 Project Structure

```
novamart-rag-pipeline/
├── src/
│   ├── generate_data.py              # Synthetic data generator (25 stores, 12 weeks)
│   ├── ingest.py                     # Chunking + embedding + ChromaDB ingestion
│   ├── query.py                      # Retrieval logic + question classification
│   └── slack_bot.py                  # Slack bot interface
├── data/
│   ├── store_master.csv
│   ├── weekly_sales.csv
│   ├── inventory.csv
│   ├── labor.csv
│   └── customer.csv
├── novamart_rag_validation.ipynb     # Full bug-finding + validation walkthrough
├── .env.example                      # Template for required API keys
└── .gitignore                        # Excludes API keys, venvs, and vector store binaries
```

---

## 📓 Start Here: The Validation Notebook

**[`novamart_rag_validation.ipynb`](./novamart_rag_validation.ipynb)** is the most important file in this repo. It documents:

- A real retrieval bug where broad, comparative questions returned confidently wrong answers
- The fix (question classification + structured retrieval)
- A second, more subtle bug found during re-validation
- Every claim from five test questions checked against raw source data — including one small error that slipped through even after both fixes

If you only look at one file, look at that one.

---

## ⚙️ Setup

**1. Clone the repo**
```
git clone https://github.com/yelsakkary1/novamart-rag-pipeline.git
cd novamart-rag-pipeline
```

**2. Create a virtual environment**
```
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```
pip install chromadb openai anthropic slack_bolt python-dotenv pandas numpy jupyter
```

**4. Add your API keys**

Copy `.env.example` to `.env` and fill in your actual keys:
```
cp .env.example .env
```

You'll need:
- `OPENAI_API_KEY` — for embeddings
- `ANTHROPIC_API_KEY` — for Claude
- `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` — for the Slack bot (requires a Slack app with Socket Mode enabled and the `reactions:write` scope under **Bot Token Scopes**)

**5. Generate the data and build the vector store**
```
python src/generate_data.py
python src/ingest.py
```

**6. Run the Slack bot**
```
python src/slack_bot.py
```

---

## ⚠️ Important Notes

- All data in this repo is **synthetic (fake)** — generated specifically to include a few intentionally underperforming stores, so the pipeline has something real to surface.
- Never commit your `.env` file — it's excluded via `.gitignore`.
- The question classifier in `query.py` is keyword-based, which is fast and free but has a real limitation: it can miss phrasings it wasn't written to catch. The validation notebook documents exactly where this happened and how it was caught.

---

## 🗺️ Roadmap

- [ ] Replace keyword-based question classification with a lightweight LLM-based intent classifier
- [ ] Add a fine-tuned model comparison alongside the RAG approach
- [ ] Expand structured retrieval to support multi-week trend questions, not just most-recent-week snapshots

---

*Built with ChromaDB · OpenAI Embeddings · Claude · Slack Bolt*
