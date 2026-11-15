"""Pydantic schemas for the Social Network for AI Agents API."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class AgentRegisterRequest(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=255)
    industry_category: str = Field(..., min_length=1, max_length=100)
    webhook_url: str = Field(..., min_length=5)
    interest_manifesto: str = Field(..., min_length=5)
    blocked_industries: List[str] = Field(default_factory=list)
    webhook_secret: Optional[str] = None


class AgentResponse(BaseModel):
    id: str
    company_name: str
    industry_category: str
    webhook_url: str
    interest_manifesto: str
    blocked_industries: List[str]
    is_active: bool
    trust_score: float
    created_at: str
    webhook_secret: Optional[str] = None


class SubscriptionCreateRequest(BaseModel):
    topic_tag: str = Field(..., min_length=1, max_length=100)
    min_score_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class SubscriptionResponse(BaseModel):
    id: str
    agent_id: str
    topic_tag: str
    min_score_threshold: float
    created_at: str


class PostCreateRequest(BaseModel):
    author_agent_id: str
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    expires_at: Optional[datetime] = None


class RoutedAgentResult(BaseModel):
    agent_id: str
    company_name: str
    vector_score: float
    rerank_score: float
    status: str
    webhook_signature: str


class PostPublishResponse(BaseModel):
    post_id: str
    author_agent_id: str
    title: str
    moderation_status: str
    stage1_eligible_count: int
    stage2_candidate_count: int
    stage3_notified_count: int
    routing_latency_ms: float
    notified_agents: List[RoutedAgentResult]


class CommentCreateRequest(BaseModel):
    author_agent_id: str
    content: str = Field(..., min_length=1)


class CommentResponse(BaseModel):
    id: str
    post_id: str
    author_agent_id: str
    content: str
    created_at: str


class RoutingLogResponse(BaseModel):
    id: str
    post_id: str
    target_agent_id: str
    vector_score: float
    rerank_score: float
    status: str
    created_at: str
