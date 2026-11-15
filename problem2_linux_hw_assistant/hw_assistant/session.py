"""Multi-Turn Conversation State & Hardware Context Cache."""

from typing import Any, Dict, List, Optional


class SessionMemory:
    """Maintains conversation turns and caches gathered hardware JSON blobs."""

    def __init__(self) -> None:
        self.history: List[Dict[str, str]] = []
        self.hardware_cache: Dict[str, Dict[str, Any]] = {}

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def get_cached(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return self.hardware_cache.get(tool_name)

    def set_cached(self, tool_name: str, data: Dict[str, Any]) -> None:
        self.hardware_cache[tool_name] = data
