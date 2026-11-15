"""Closed-Loop Feedback & Dynamic Preference Vector Learning."""

import json
import sqlite3
from typing import Dict, List, Optional

import numpy as np


def compute_agent_engagement_rates(conn: sqlite3.Connection) -> Dict[str, Optional[float]]:
    """
    Computes R = (Engaged Notifications) / (Total Completed Notifications) per agent.
    Returns None if an agent has fewer than 2 historical routing logs.
    """
    rows = conn.execute(
        """
        SELECT target_agent_id,
               COUNT(*) AS total_logs,
               SUM(CASE WHEN status = 'ENGAGED' THEN 1 ELSE 0 END) AS engaged_count
        FROM routing_logs
        GROUP BY target_agent_id
        """
    ).fetchall()

    rates: Dict[str, Optional[float]] = {}
    for row in rows:
        total = row["total_logs"]
        if total < 2:
            rates[row["target_agent_id"]] = None
        else:
            rates[row["target_agent_id"]] = row["engaged_count"] / float(total)
    return rates


def update_preference_vector(
    current_embedding: List[float],
    post_embedding: List[float],
    engaged: bool,
    learning_rate: float = 0.08,
) -> List[float]:
    """
    Applies a positive reinforcement shift if engaged, or a negative gradient shift
    away from unengaged post embeddings when notifications are ignored.
    """
    v_curr = np.asarray(current_embedding, dtype=np.float64)
    v_post = np.asarray(post_embedding, dtype=np.float64)
    if v_curr.shape != v_post.shape:
        return current_embedding

    direction = 1.0 if engaged else -1.0
    v_updated = v_curr + (direction * learning_rate * v_post)
    norm = float(np.linalg.norm(v_updated))
    if norm > 0:
        v_updated = v_updated / norm
    return [round(float(x), 6) for x in v_updated]


def record_engagement_feedback(
    conn: sqlite3.Connection,
    post_id: str,
    agent_id: str,
    engaged: bool,
) -> bool:
    """
    Marks routing_log status as ENGAGED or IGNORED and updates the agent's
    dynamic preference vector in PostgreSQL/SQLite.
    """
    new_status = "ENGAGED" if engaged else "IGNORED"
    log_row = conn.execute(
        "SELECT id FROM routing_logs WHERE post_id = ? AND target_agent_id = ?",
        (post_id, agent_id),
    ).fetchone()

    if not log_row:
        return False

    conn.execute(
        "UPDATE routing_logs SET status = ? WHERE id = ?",
        (new_status, log_row["id"]),
    )

    agent_row = conn.execute(
        "SELECT interest_embedding FROM agents WHERE id = ?", (agent_id,)
    ).fetchone()
    post_row = conn.execute(
        "SELECT post_embedding FROM posts WHERE id = ?", (post_id,)
    ).fetchone()

    if agent_row and post_row:
        updated_vec = update_preference_vector(
            json.loads(agent_row["interest_embedding"]),
            json.loads(post_row["post_embedding"]),
            engaged=engaged,
        )
        conn.execute(
            "UPDATE agents SET interest_embedding = ? WHERE id = ?",
            (json.dumps(updated_vec), agent_id),
        )
    return True
