"""High-level Hardware Inspection Tools with Root Fallback & OpenAI/Pydantic Tool Schemas."""

import os
from typing import Any, Dict, List

from hw_assistant.parsers import (
    parse_dmidecode_memory,
    parse_free_bytes,
    parse_lsblk_json,
    parse_lscpu,
    parse_lspci_gpu,
    parse_proc_meminfo,
)
from hw_assistant.security import discover_available_tools, execute_allowlisted_command


def get_tool_schemas(available_tools: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Builds OpenAI-compatible function-calling schemas only for available tools."""
    schemas: List[Dict[str, Any]] = [
        {
            "name": "get_memory_topology",
            "description": (
                "Retrieves physical RAM details including installed capacity, total slots, "
                "available slots, speed, and maximum supported motherboard capacity."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "include_empty_slots": {
                        "type": "BOOLEAN",
                        "description": "Set to true to return details on unused memory slots.",
                    }
                },
                "required": [],
            },
        }
    ]
    for tool_id in ("get_cpu_info", "get_block_devices", "get_gpu_info"):
        if available_tools.get(tool_id, {}).get("available"):
            schemas.append(
                {
                    "name": tool_id,
                    "description": available_tools[tool_id]["description"],
                    "parameters": {"type": "OBJECT", "properties": {}, "required": []},
                }
            )
    return schemas


def run_hardware_tool(tool_name: str, available_tools: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dispatches a validated tool call and returns normalized JSON."""
    registry = available_tools if available_tools is not None else discover_available_tools()

    if tool_name == "get_cpu_info":
        res = execute_allowlisted_command("get_cpu_info", registry)
        if not res["ok"]:
            return {"error": res["error"], "available": False}
        return {"available": True, **parse_lscpu(res["stdout"])}

    if tool_name == "get_block_devices":
        res = execute_allowlisted_command("get_block_devices", registry)
        if not res["ok"]:
            return {"error": res["error"], "available": False}
        return {"available": True, **parse_lsblk_json(res["stdout"])}

    if tool_name == "get_gpu_info":
        res = execute_allowlisted_command("get_gpu_info", registry)
        if not res["ok"]:
            return {"error": res["error"], "available": False}
        return {"available": True, **parse_lspci_gpu(res["stdout"])}

    if tool_name == "get_memory_topology":
        # 1. Try scoped non-interactive sudo dmidecode first (if installed)
        dmi_data: Dict[str, Any] = {"dmi_available": False}
        if registry.get("get_dmi_memory", {}).get("available"):
            is_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
            dmi_res = execute_allowlisted_command("get_dmi_memory", registry, use_sudo=not is_root)
            if dmi_res["ok"]:
                dmi_data = parse_dmidecode_memory(dmi_res["stdout"])
            else:
                dmi_data["dmi_fallback_reason"] = dmi_res["error"]

        # 2. Always gather unprivileged RAM summary via `free -b` or `/proc/meminfo`
        mem_summary: Dict[str, Any] = {}
        free_res = execute_allowlisted_command("get_memory_summary", registry)
        if free_res["ok"]:
            mem_summary = parse_free_bytes(free_res["stdout"])
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                mem_summary = parse_proc_meminfo(fh.read())

        return {
            "available": bool(dmi_data.get("dmi_available") or mem_summary.get("total_gb") is not None),
            **mem_summary,
            **dmi_data,
        }

    raise ValueError(f"Unknown hardware tool: {tool_name}")
