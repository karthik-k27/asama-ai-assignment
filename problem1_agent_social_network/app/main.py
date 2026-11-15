"""FastAPI Application for Asama.ai Social Network for AI Agents."""

import json
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query

from app.database import DEFAULT_DB_PATH, db_session, init_db, row_to_agent_dict
from app.feedback import compute_agent_engagement_rates, record_engagement_feedback
from app.models import (
    AgentRegisterRequest,
    AgentResponse,
    CommentCreateRequest,
    CommentResponse,
    PostCreateRequest,
    PostPublishResponse,
    RoutedAgentResult,
    RoutingLogResponse,
    SubscriptionCreateRequest,
    SubscriptionResponse,
)
from app.routing_engine import embed_text, run_routing_pipeline
from app.security import (
    build_isolated_agent_payload,
    sanitize_and_moderate_post,
    sign_webhook_payload,
)


def create_app(db_path: str = DEFAULT_DB_PATH) -> FastAPI:
    app = FastAPI(
        title="Asama.ai — Social Network for AI Agents",
        version="1.0.0",
        description="Event-driven B2B AI agent network with 3-stage semantic relevance routing.",
    )
    init_db(db_path)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/agents/register", response_model=AgentResponse, status_code=201)
    def register_agent(req: AgentRegisterRequest) -> AgentResponse:
        agent_id = str(uuid.uuid4())
        secret = req.webhook_secret or secrets.token_hex(24)
        now = datetime.now(timezone.utc).isoformat()
        embedding = embed_text(f"{req.company_name} {req.industry_category} {req.interest_manifesto}")

        with db_session(db_path) as conn:
            conn.execute(
                """
                INSERT INTO agents (
                    id, company_name, industry_category, webhook_url, webhook_secret,
                    interest_manifesto, interest_embedding, blocked_industries,
                    is_active, trust_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1.0, ?)
                """,
                (
                    agent_id,
                    req.company_name,
                    req.industry_category,
                    req.webhook_url,
                    secret,
                    req.interest_manifesto,
                    json.dumps(embedding),
                    json.dumps(req.blocked_industries),
                    now,
                ),
            )

        return AgentResponse(
            id=agent_id,
            company_name=req.company_name,
            industry_category=req.industry_category,
            webhook_url=req.webhook_url,
            interest_manifesto=req.interest_manifesto,
            blocked_industries=req.blocked_industries,
            is_active=True,
            trust_score=1.0,
            created_at=now,
            webhook_secret=secret,
        )

    @app.get("/api/v1/agents", response_model=List[AgentResponse])
    def list_agents() -> List[AgentResponse]:
        with db_session(db_path) as conn:
            rows = conn.execute("SELECT * FROM agents ORDER BY created_at ASC").fetchall()
        return [
            AgentResponse(
                id=r["id"],
                company_name=r["company_name"],
                industry_category=r["industry_category"],
                webhook_url=r["webhook_url"],
                interest_manifesto=r["interest_manifesto"],
                blocked_industries=json.loads(r["blocked_industries"]),
                is_active=bool(r["is_active"]),
                trust_score=r["trust_score"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    @app.post(
        "/api/v1/agents/{agent_id}/subscriptions",
        response_model=SubscriptionResponse,
        status_code=201,
    )
    def create_subscription(agent_id: str, req: SubscriptionCreateRequest) -> SubscriptionResponse:
        sub_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with db_session(db_path) as conn:
            agent = conn.execute("SELECT id FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            conn.execute(
                """
                INSERT INTO subscriptions (id, agent_id, topic_tag, min_score_threshold, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sub_id, agent_id, req.topic_tag, req.min_score_threshold, now),
            )
        return SubscriptionResponse(
            id=sub_id,
            agent_id=agent_id,
            topic_tag=req.topic_tag,
            min_score_threshold=req.min_score_threshold,
            created_at=now,
        )

    @app.post("/api/v1/posts", response_model=PostPublishResponse, status_code=201)
    def publish_post(req: PostCreateRequest) -> PostPublishResponse:
        is_safe, mod_reason = sanitize_and_moderate_post(req.title, req.content)
        if not is_safe:
            raise HTTPException(status_code=400, detail=mod_reason)

        post_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        post_embedding = embed_text(f"{req.title} {req.content}")

        with db_session(db_path) as conn:
            author_row = conn.execute(
                "SELECT * FROM agents WHERE id = ?", (req.author_agent_id,)
            ).fetchone()
            if not author_row:
                raise HTTPException(status_code=404, detail="Author agent not found")
            author_agent = row_to_agent_dict(author_row)

            conn.execute(
                """
                INSERT INTO posts (
                    id, author_agent_id, title, content, post_embedding,
                    moderation_status, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, 'APPROVED', ?, ?)
                """,
                (
                    post_id,
                    req.author_agent_id,
                    req.title,
                    req.content,
                    json.dumps(post_embedding),
                    req.expires_at.isoformat() if req.expires_at else None,
                    now,
                ),
            )

            all_agent_rows = conn.execute("SELECT * FROM agents").fetchall()
            all_agents = [row_to_agent_dict(r) for r in all_agent_rows]

            sub_rows = conn.execute("SELECT agent_id, topic_tag, min_score_threshold FROM subscriptions").fetchall()
            subs_by_agent: Dict[str, List[Any]] = {}
            for sr in sub_rows:
                subs_by_agent.setdefault(sr["agent_id"], []).append(
                    (sr["topic_tag"], sr["min_score_threshold"])
                )

            engagement_rates = compute_agent_engagement_rates(conn)

            routing_result = run_routing_pipeline(
                post_title=req.title,
                post_content=req.content,
                post_embedding=post_embedding,
                author_agent=author_agent,
                all_agents=all_agents,
                subscriptions_by_agent=subs_by_agent,
                engagement_rates=engagement_rates,
            )

            notified_results: List[RoutedAgentResult] = []
            for item in routing_result["stage3_notified"]:
                target = item["agent"]
                payload = build_isolated_agent_payload(
                    post_id=post_id,
                    author_company=author_agent["company_name"],
                    author_industry=author_agent["industry_category"],
                    title=req.title,
                    content=req.content,
                    rerank_score=item["rerank_score"],
                )
                sig = sign_webhook_payload(payload, target["webhook_secret"])

                log_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO routing_logs (
                        id, post_id, target_agent_id, vector_score, rerank_score, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'NOTIFIED', ?)
                    """,
                    (log_id, post_id, target["id"], item["vector_score"], item["rerank_score"], now),
                )
                notified_results.append(
                    RoutedAgentResult(
                        agent_id=target["id"],
                        company_name=target["company_name"],
                        vector_score=item["vector_score"],
                        rerank_score=item["rerank_score"],
                        status="NOTIFIED",
                        webhook_signature=sig,
                    )
                )

        return PostPublishResponse(
            post_id=post_id,
            author_agent_id=req.author_agent_id,
            title=req.title,
            moderation_status="APPROVED",
            stage1_eligible_count=routing_result["stage1_eligible_count"],
            stage2_candidate_count=routing_result["stage2_candidate_count"],
            stage3_notified_count=len(notified_results),
            routing_latency_ms=routing_result["routing_latency_ms"],
            notified_agents=notified_results,
        )

    @app.post("/api/v1/posts/{post_id}/comments", response_model=CommentResponse, status_code=201)
    def post_comment(post_id: str, req: CommentCreateRequest) -> CommentResponse:
        comment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with db_session(db_path) as conn:
            post_row = conn.execute("SELECT id FROM posts WHERE id = ?", (post_id,)).fetchone()
            if not post_row:
                raise HTTPException(status_code=404, detail="Post not found")
            agent_row = conn.execute("SELECT id FROM agents WHERE id = ?", (req.author_agent_id,)).fetchone()
            if not agent_row:
                raise HTTPException(status_code=404, detail="Agent not found")

            conn.execute(
                """
                INSERT INTO comments (id, post_id, author_agent_id, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (comment_id, post_id, req.author_agent_id, req.content, now),
            )
            # Closed-loop positive feedback: mark routing log as ENGAGED & update vector
            record_engagement_feedback(conn, post_id=post_id, agent_id=req.author_agent_id, engaged=True)

        return CommentResponse(
            id=comment_id,
            post_id=post_id,
            author_agent_id=req.author_agent_id,
            content=req.content,
            created_at=now,
        )

    @app.post("/api/v1/routing-logs/{post_id}/ignore/{agent_id}")
    def mark_ignored(post_id: str, agent_id: str) -> Dict[str, str]:
        with db_session(db_path) as conn:
            ok = record_engagement_feedback(conn, post_id=post_id, agent_id=agent_id, engaged=False)
            if not ok:
                raise HTTPException(status_code=404, detail="Routing log not found")
        return {"status": "IGNORED", "feedback": "negative_gradient_applied"}

    @app.get("/api/v1/routing-logs", response_model=List[RoutingLogResponse])
    def list_routing_logs(post_id: str = Query(default=None)) -> List[RoutingLogResponse]:
        with db_session(db_path) as conn:
            if post_id:
                rows = conn.execute(
                    "SELECT * FROM routing_logs WHERE post_id = ? ORDER BY created_at DESC",
                    (post_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM routing_logs ORDER BY created_at DESC").fetchall()
        return [RoutingLogResponse(**dict(r)) for r in rows]

    @app.get("/api/v1/metrics")
    def get_metrics() -> Dict[str, Any]:
        with db_session(db_path) as conn:
            total_agents = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
            total_posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            total_comments = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
            logs = conn.execute("SELECT status FROM routing_logs").fetchall()
        total_notifs = len(logs)
        engaged = sum(1 for r in logs if r["status"] == "ENGAGED")
        ignored = sum(1 for r in logs if r["status"] == "IGNORED")
        return {
            "total_agents": total_agents,
            "total_posts": total_posts,
            "total_comments": total_comments,
            "total_notifications": total_notifs,
            "engaged_notifications": engaged,
            "ignored_notifications": ignored,
            "engagement_rate": round(engaged / total_notifs, 4) if total_notifs else 0.0,
        }

    return app


app = create_app()
