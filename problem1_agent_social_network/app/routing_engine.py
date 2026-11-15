"""Multi-Stage Semantic Relevance Routing Engine (SQL Guardrails -> ANN Vector -> Reranker)."""

import hashlib
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Domain semantic concept clusters for zero-API local embeddings.
# In production, swap `embed_text()` with OpenAI `text-embedding-3-small`
# or a local sentence-transformer model.
CONCEPT_CLUSTERS: Dict[str, List[str]] = {
    "payments_fintech": [
        "upi", "payment", "payments", "cashback", "wallet", "card", "credit",
        "debit", "bank", "banking", "fintech", "checkout", "reward", "rewards",
        "emi", "hdfc", "phonepe", "paytm", "rupay", "visa", "mastercard",
    ],
    "food_delivery": [
        "food", "delivery", "restaurant", "dining", "meal", "order", "orders",
        "swiggy", "zomato", "grocery", "instamart", "snack", "cafe",
    ],
    "promotions_festive": [
        "offer", "offers", "discount", "discounts", "promo", "promotion",
        "promotions", "deal", "deals", "sale", "festive", "diwali", "holi",
        "flat", "off", "cashback", "coupon", "voucher", "partnership", "partnerships",
    ],
    "ecommerce_retail": [
        "ecommerce", "e-commerce", "shopping", "retail", "marketplace",
        "flipkart", "amazon", "fashion", "apparel", "electronics", "cart",
    ],
    "automotive_mobility": [
        "auto", "automotive", "bike", "motorcycle", "scooter", "car", "ev",
        "vehicle", "bajaj", "two-wheeler", "dealer", "showroom", "service",
        "spare", "engine", "mileage",
    ],
    "travel_hospitality": [
        "travel", "flight", "flights", "hotel", "hotels", "booking", "trip",
        "cab", "airline", "airport", "vacation", "tourism",
    ],
    "b2b_enterprise_saas": [
        "saas", "cloud", "api", "enterprise", "b2b", "software", "analytics",
        "ai", "agent", "automation", "crm", "erp",
    ],
}

EMBEDDING_DIM = 128


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\-]+", text.lower())


def embed_text(text: str) -> List[float]:
    """
    Produces a normalized 128-dim semantic+lexical embedding.
    Uses OpenAI `text-embedding-3-small` if OPENAI_API_KEY is set and `ASAMA_USE_OPENAI=1`;
    otherwise uses deterministic local semantic-cluster + hashed-ngram projection.
    """
    if os.environ.get("ASAMA_USE_OPENAI") == "1" and os.environ.get("OPENAI_API_KEY"):
        try:
            import httpx

            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={"model": "text-embedding-3-small", "input": text},
                timeout=5.0,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception:
            pass  # Fall back to deterministic local embedding

    vec = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    tokens = _tokenize(text)
    token_set = set(tokens)

    # First 32 dims: weighted semantic concept cluster activations
    for idx, (_cluster_name, keywords) in enumerate(CONCEPT_CLUSTERS.items()):
        overlap = sum(1 for kw in keywords if kw in token_set)
        if overlap > 0:
            # Sub-linear TF saturation
            vec[idx * 4 : idx * 4 + 4] += 1.0 + math.log1p(overlap)

    # Remaining dims: deterministic feature-hashed unigrams & bigrams
    hash_start = 32
    hash_span = EMBEDDING_DIM - hash_start
    ngrams = list(tokens) + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    for gram in ngrams:
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        bucket = hash_start + (h % hash_span)
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[bucket] += sign * 0.35

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return [round(float(x), 6) for x in vec]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    score = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, score))


def stage1_sql_guardrails(
    agents: List[Dict[str, Any]],
    author_agent: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Stage 1: Fast SQL/metadata guardrails.
    Excludes deactivated agents, the publishing agent itself, low-trust agents,
    and explicitly blocked industries.
    """
    author_id = author_agent["id"]
    author_industry = author_agent["industry_category"].lower()
    author_blocked = {b.lower() for b in author_agent.get("blocked_industries", [])}

    eligible: List[Dict[str, Any]] = []
    for agent in agents:
        if not agent.get("is_active", True):
            continue
        if agent["id"] == author_id:
            continue
        if agent.get("trust_score", 1.0) < 0.3:
            continue
        agent_industry = agent["industry_category"].lower()
        agent_blocked = {b.lower() for b in agent.get("blocked_industries", [])}
        if author_industry in agent_blocked or agent_industry in author_blocked:
            continue
        eligible.append(agent)
    return eligible


def stage2_vector_retrieval(
    post_embedding: List[float],
    eligible_agents: List[Dict[str, Any]],
    top_k: int = 50,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Stage 2: Dense Vector ANN Search (Cosine Similarity, Top K=50).
    In PostgreSQL+pgvector, this runs via HNSW index `<=>` operator.
    """
    scored = [
        (agent, cosine_similarity(post_embedding, agent["interest_embedding"]))
        for agent in eligible_agents
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _compute_cross_encoder_score(
    post_title: str,
    post_content: str,
    author_industry: str,
    agent: Dict[str, Any],
    vector_score: float,
    subscriptions: List[str],
    engagement_rate: Optional[float],
) -> float:
    """
    Stage 3: Cross-Encoder / Fast LLM Judge simulation.
    Evaluates cross-domain synergy (e.g., Food Delivery Offer <-> Payments/Cards),
    subscription tag matches, trust score, and historical engagement rate.
    """
    post_tokens = set(_tokenize(f"{post_title} {post_content} {author_industry}"))
    manifesto_tokens = set(_tokenize(f"{agent['interest_manifesto']} {agent['industry_category']}"))

    # Concept-level synergy check
    post_clusters = {
        cname for cname, kws in CONCEPT_CLUSTERS.items() if any(k in post_tokens for k in kws)
    }
    agent_clusters = {
        cname for cname, kws in CONCEPT_CLUSTERS.items() if any(k in manifesto_tokens for k in kws)
    }

    cluster_overlap = len(post_clusters & agent_clusters) / max(1, len(post_clusters))

    # Cross-industry partnership synergy boost (e.g., Food/E-commerce Promo <-> Payments/Fintech)
    synergy_boost = 0.0
    if "promotions_festive" in post_clusters and (
        "payments_fintech" in agent_clusters and ("food_delivery" in post_clusters or "ecommerce_retail" in post_clusters)
    ):
        synergy_boost = 0.20

    # Subscription tag match boost
    sub_boost = 0.0
    for tag in subscriptions:
        tag_tokens = set(_tokenize(tag))
        if tag_tokens & post_tokens:
            sub_boost = 0.12
            break

    # Historical engagement adjustment (closed-loop feedback)
    feedback_adj = 0.0
    if engagement_rate is not None:
        # Neutral at 0.5; +0.05 if highly engaged, -0.08 if consistently ignored
        feedback_adj = (engagement_rate - 0.5) * 0.12

    raw_score = (0.55 * vector_score) + (0.30 * cluster_overlap) + synergy_boost + sub_boost + feedback_adj
    trust_factor = min(1.0, max(0.5, agent.get("trust_score", 1.0)))
    final_score = max(0.0, min(1.0, raw_score * trust_factor))
    return round(final_score, 4)


def run_routing_pipeline(
    post_title: str,
    post_content: str,
    post_embedding: List[float],
    author_agent: Dict[str, Any],
    all_agents: List[Dict[str, Any]],
    subscriptions_by_agent: Dict[str, List[Tuple[str, float]]],
    engagement_rates: Dict[str, Optional[float]],
    top_k_stage2: int = 50,
    max_notify_stage3: int = 5,
    default_threshold: float = 0.75,
) -> Dict[str, Any]:
    """Executes the full 3-tier candidate filtering pipeline and measures latency."""
    t0 = time.perf_counter()

    # Stage 1: Hard SQL Guardrails
    eligible = stage1_sql_guardrails(all_agents, author_agent)

    # Stage 2: Dense Vector Search
    candidates = stage2_vector_retrieval(post_embedding, eligible, top_k=top_k_stage2)

    # Stage 3: Reranker & Threshold Gating
    reranked: List[Dict[str, Any]] = []
    for agent, vec_score in candidates:
        aid = agent["id"]
        agent_subs = subscriptions_by_agent.get(aid, [])
        sub_tags = [tag for tag, _ in agent_subs]
        threshold = min([thresh for _, thresh in agent_subs], default=default_threshold)

        rerank_score = _compute_cross_encoder_score(
            post_title=post_title,
            post_content=post_content,
            author_industry=author_agent["industry_category"],
            agent=agent,
            vector_score=vec_score,
            subscriptions=sub_tags,
            engagement_rate=engagement_rates.get(aid),
        )
        if rerank_score >= threshold:
            reranked.append(
                {
                    "agent": agent,
                    "vector_score": round(vec_score, 4),
                    "rerank_score": rerank_score,
                }
            )

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    final_notified = reranked[:max_notify_stage3]
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "stage1_eligible_count": len(eligible),
        "stage2_candidate_count": len(candidates),
        "stage3_notified": final_notified,
        "routing_latency_ms": latency_ms,
    }
