"""Intent Router & Strictly Grounded Answer Synthesizer (Ollama / OpenAI / Offline)."""

import json
import os
import re
from typing import Any, Dict, List, Optional

from hw_assistant.security import discover_available_tools
from hw_assistant.session import SessionMemory
from hw_assistant.tools import run_hardware_tool


SYSTEM_PROMPT = (
    "You are a local Linux Hardware Q&A Assistant. "
    "Base your response SOLELY on the provided JSON tool output. "
    "If the JSON does not contain the answer or a tool is unavailable, state: "
    "'I cannot determine this hardware detail from system data' and list the missing data/tool. "
    "Never guess or extrapolate hardware specifications."
)


def select_tools_for_query(query: str) -> List[str]:
    """
    Deterministic intent-to-tool router.
    In production with Ollama/OpenAI, this acts as the fast pre-router or fallback.
    """
    q = query.lower()
    selected: List[str] = []
    if any(k in q for k in ("ram", "memory", "dimm", "slot", "upgrade")):
        selected.append("get_memory_topology")
    if any(k in q for k in ("cpu", "processor", "core", "thread", "architecture", "virtualization")):
        selected.append("get_cpu_info")
    if any(k in q for k in ("ssd", "nvme", "disk", "storage", "drive", "hdd")):
        selected.append("get_block_devices")
    if any(k in q for k in ("gpu", "graphics", "vga", "nvidia", "amd", "display")):
        selected.append("get_gpu_info")
    return selected


def synthesize_grounded_answer(query: str, gathered: Dict[str, Dict[str, Any]]) -> str:
    """
    Synthesizes a clear, plain-language answer strictly from verified JSON tool outputs.
    Optionally delegates to local Ollama (`OLLAMA_MODEL`) if `ASAMA_USE_OLLAMA=1`.
    """
    if os.environ.get("ASAMA_USE_OLLAMA") == "1":
        try:
            import httpx

            model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
            resp = httpx.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Question: {query}\n\nVerified JSON:\n{json.dumps(gathered, indent=2)}",
                        },
                    ],
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception:
            pass  # Fall back to deterministic grounded synthesis

    if not gathered:
        return (
            "I cannot determine this hardware detail from system data. "
            "Try asking about RAM/memory, CPU, storage/NVMe drives, or GPU."
        )

    parts: List[str] = []

    if "get_memory_topology" in gathered:
        mem = gathered["get_memory_topology"]
        total_gb = mem.get("total_gb")
        if mem.get("dmi_available"):
            used = mem.get("used_slots", 0)
            total_slots = mem.get("total_slots", 0)
            free_slots = mem.get("free_slots", 0)
            max_cap = mem.get("maximum_capacity", "Data unavailable")
            modules_desc = ", ".join(
                f"{m['size']} {m['type']} ({m['speed']})" for m in mem.get("installed_modules", [])
            )
            parts.append(
                f"Installed RAM: {total_gb} GB across {used} of {total_slots} slots "
                f"({free_slots} free). Motherboard maximum supported capacity: {max_cap}."
                + (f" Modules: {modules_desc}." if modules_desc else "")
            )
        elif total_gb is not None:
            avail_gb = mem.get("available_gb")
            parts.append(
                f"Installed RAM: {total_gb} GB (currently {avail_gb} GB available). "
                "Slot count and maximum supported motherboard capacity cannot be determined "
                "from system data without root privileges for `dmidecode`."
            )
        else:
            parts.append("I cannot determine RAM details from system data.")

    if "get_cpu_info" in gathered:
        cpu = gathered["get_cpu_info"]
        if cpu.get("available"):
            parts.append(
                f"CPU: {cpu.get('model_name', 'Unknown')} ({cpu.get('architecture', 'Unknown')}), "
                f"{cpu.get('cpus', 'Unknown')} logical CPUs."
            )
        else:
            parts.append("I cannot determine CPU details from system data (lscpu unavailable).")

    if "get_block_devices" in gathered:
        blk = gathered["get_block_devices"]
        if blk.get("available") and blk.get("disks"):
            disk_strs = [
                f"{d['name']} ({d['size']}, transport={d['transport']}, nvme={d['is_nvme']})"
                for d in blk["disks"]
            ]
            parts.append("Storage disks: " + "; ".join(disk_strs) + ".")
        else:
            parts.append("I cannot determine storage drive details from system data.")

    if "get_gpu_info" in gathered:
        gpu = gathered["get_gpu_info"]
        if gpu.get("available") and gpu.get("gpus"):
            parts.append("GPU(s): " + " | ".join(gpu["gpus"]))
        elif gpu.get("available"):
            parts.append("No discrete or integrated PCIe GPU controller was reported by lspci.")
        else:
            parts.append("I cannot determine GPU details from system data (lspci unavailable).")

    return " ".join(parts)


class HardwareAssistant:
    """Orchestrates multi-turn hardware Q&A using cached state and allow-listed tools."""

    def __init__(self, tool_runner=None) -> None:
        self.available_tools = discover_available_tools()
        self.session = SessionMemory()
        self._tool_runner = tool_runner or (lambda name: run_hardware_tool(name, self.available_tools))

    def ask(self, query: str) -> Dict[str, Any]:
        self.session.add_turn("user", query)
        needed_tools = select_tools_for_query(query)

        gathered: Dict[str, Dict[str, Any]] = {}
        tools_executed: List[str] = []
        cache_hits: List[str] = []

        for tool_name in needed_tools:
            cached = self.session.get_cached(tool_name)
            if cached is not None:
                gathered[tool_name] = cached
                cache_hits.append(tool_name)
            else:
                result = self._tool_runner(tool_name)
                self.session.set_cached(tool_name, result)
                gathered[tool_name] = result
                tools_executed.append(tool_name)

        answer = synthesize_grounded_answer(query, gathered)
        self.session.add_turn("assistant", answer)
        return {
            "answer": answer,
            "tools_executed": tools_executed,
            "cache_hits": cache_hits,
            "raw_data": gathered,
        }
