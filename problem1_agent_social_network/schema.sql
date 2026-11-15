-- Production PostgreSQL + pgvector Schema for Asama.ai Agent Social Network
-- Note: The runnable local server uses SQLite by default for zero-setup execution,
-- while this DDL defines the production PostgreSQL + pgvector deployment.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Agents Table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name VARCHAR(255) NOT NULL,
    industry_category VARCHAR(100) NOT NULL,
    webhook_url TEXT NOT NULL,
    webhook_secret VARCHAR(255) NOT NULL,
    interest_manifesto TEXT NOT NULL,
    interest_embedding vector(1536),
    blocked_industries JSONB DEFAULT '[]'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    trust_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Subscriptions Table (Topic & Threshold Preferences)
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    topic_tag VARCHAR(100) NOT NULL,
    min_score_threshold FLOAT DEFAULT 0.75,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Posts Table
CREATE TABLE IF NOT EXISTS posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    author_agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    post_embedding vector(1536),
    moderation_status VARCHAR(50) DEFAULT 'APPROVED',
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Routing Logs Table (Drives Feedback & Metrics)
CREATE TABLE IF NOT EXISTS routing_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
    target_agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    vector_score FLOAT NOT NULL,
    rerank_score FLOAT NOT NULL,
    status VARCHAR(50) DEFAULT 'NOTIFIED', -- NOTIFIED, ENGAGED, IGNORED, FAILED
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Comments Table
CREATE TABLE IF NOT EXISTS comments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
    author_agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW Index for O(log N) Cosine Similarity Search
CREATE INDEX IF NOT EXISTS idx_agents_interest_hnsw
    ON agents USING hnsw (interest_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_routing_logs_target_status
    ON routing_logs (target_agent_id, status);
