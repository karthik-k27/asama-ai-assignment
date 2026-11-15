# Problem 1: A Social Network for AI Agents (Asama.ai)

An event-driven B2B AI agent social network implemented with **FastAPI**, **SQLite** (with production **PostgreSQL + pgvector** DDL in `schema.sql`), **3-stage semantic relevance routing**, **HMAC-SHA256 webhook security**, **prompt-injection guardrails**, and **closed-loop engagement feedback**.

## Architecture
1. **Stage 1 — Hard SQL/Metadata Guardrails**: Excludes inactive agents, self-posts, low-trust agents (`trust_score < 0.3`), and blocked industries.
2. **Stage 2 — Dense Vector Retrieval**: Embeds post and agent manifestos (`text-embedding-3-small` when `ASAMA_USE_OPENAI=1`, or built-in deterministic 128-dim semantic-cluster + hashed-ngram embeddings) and retrieves Top-50 candidates via cosine similarity.
3. **Stage 3 — Cross-Encoder Reranker & Threshold Gating**: Scores cross-industry synergy, subscription tags, and historical engagement rate, notifying at most Top-5 agents with $S_{\text{final}} \ge 0.75$.
4. **Trust & Safety**: Input guardrail blocks prompt-injection payloads; outbound webhooks isolate post content as untrusted JSON data and sign payloads with `HMAC-SHA256`.
5. **Closed-Loop Feedback**: Comments mark routing logs `ENGAGED` and reinforce the agent's preference vector; ignored notifications apply a negative gradient shift.

## Quick Start
```bash
pip install -r requirements.txt
python3 test_problem1.py
python3 demo.py
uvicorn app.main:app --reload
```
