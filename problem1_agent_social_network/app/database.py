"""SQLite persistence layer mirroring the PostgreSQL + pgvector schema."""

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional


DEFAULT_DB_PATH = os.environ.get("ASAMA_DB_PATH", "asama_network.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def db_session(db_path: str = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    with db_session(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                industry_category TEXT NOT NULL,
                webhook_url TEXT NOT NULL,
                webhook_secret TEXT NOT NULL,
                interest_manifesto TEXT NOT NULL,
                interest_embedding TEXT NOT NULL,
                blocked_industries TEXT NOT NULL DEFAULT '[]',
                is_active INTEGER NOT NULL DEFAULT 1,
                trust_score REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subscriptions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                topic_tag TEXT NOT NULL,
                min_score_threshold REAL NOT NULL DEFAULT 0.75,
                created_at TEXT NOT NULL,
                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY,
                author_agent_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                post_embedding TEXT NOT NULL,
                moderation_status TEXT NOT NULL DEFAULT 'APPROVED',
                expires_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (author_agent_id) REFERENCES agents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS routing_logs (
                id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                target_agent_id TEXT NOT NULL,
                vector_score REAL NOT NULL,
                rerank_score REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'NOTIFIED',
                created_at TEXT NOT NULL,
                FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
                FOREIGN KEY (target_agent_id) REFERENCES agents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS comments (
                id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                author_agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
                FOREIGN KEY (author_agent_id) REFERENCES agents(id) ON DELETE CASCADE
            );
            """
        )


def row_to_agent_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["interest_embedding"] = json.loads(d["interest_embedding"])
    d["blocked_industries"] = json.loads(d["blocked_industries"])
    d["is_active"] = bool(d["is_active"])
    return d
