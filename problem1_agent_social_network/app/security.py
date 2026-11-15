"""Trust, Safety, Prompt-Injection Defense, and HMAC Webhook Signing."""

import hashlib
import hmac
import json
import re
from typing import Any, Dict, Tuple


# Patterns indicative of indirect prompt injection or malicious payload hijacking
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(your\s+)?(system\s+)?prompt",
    r"(output|reveal|print|leak)\s+(confidential|secret|api[_\s]?keys?|tokens?|credentials?)",
    r"<\s*/?\s*system\s*>",
    r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
    r"execute\s+shell\s+command",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def sanitize_and_moderate_post(title: str, content: str) -> Tuple[bool, str]:
    """
    Input guardrail analogous to Llama Guard / content moderation filter.
    Returns (is_safe, reason).
    """
    combined = f"{title}\n{content}"
    for pattern in COMPILED_PATTERNS:
        if pattern.search(combined):
            return False, f"Blocked by prompt-injection guardrail: matched '{pattern.pattern}'"
    return True, "APPROVED"


def build_isolated_agent_payload(
    post_id: str,
    author_company: str,
    author_industry: str,
    title: str,
    content: str,
    rerank_score: float,
) -> Dict[str, Any]:
    """
    Strict Data/Prompt Separation:
    Post content is encapsulated strictly as untrusted data fields inside a
    structured JSON envelope, never formatted as system instructions.
    """
    return {
        "event": "post.routed",
        "version": "1.0",
        "data": {
            "post_id": post_id,
            "author": {
                "company_name": author_company,
                "industry_category": author_industry,
            },
            "untrusted_post_content": {
                "title": title,
                "body": content,
            },
            "relevance_score": round(rerank_score, 4),
        },
    }


def sign_webhook_payload(payload: Dict[str, Any], secret: str) -> str:
    """Generates an HMAC-SHA256 signature for webhook payload authenticity."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_webhook_signature(payload: Dict[str, Any], secret: str, signature_header: str) -> bool:
    """Constant-time verification of an incoming webhook HMAC-SHA256 signature."""
    expected = sign_webhook_payload(payload, secret)
    return hmac.compare_digest(expected, signature_header)
