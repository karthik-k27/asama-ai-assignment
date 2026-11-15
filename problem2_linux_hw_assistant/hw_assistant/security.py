"""Allow-List Verification, Argument Validation, and Safe Subprocess Execution (shell=False)."""

import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional


# Strict allow-list mapping internal tool IDs to static binary names and fixed arguments.
# The LLM never constructs shell strings; it only selects a tool ID from this map.
STATIC_TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "get_cpu_info": {
        "binary": "lscpu",
        "args": [],
        "requires_root": False,
        "description": "Retrieves CPU architecture, model name, core/thread counts, and virtualization support.",
    },
    "get_block_devices": {
        "binary": "lsblk",
        "args": ["-J", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL,ROTA,TRAN"],
        "requires_root": False,
        "description": "Retrieves storage drives, NVMe/SATA transport types, capacities, and mountpoints.",
    },
    "get_memory_summary": {
        "binary": "free",
        "args": ["-b"],
        "requires_root": False,
        "description": "Retrieves total, used, free, and available system RAM in bytes.",
    },
    "get_dmi_memory": {
        "binary": "dmidecode",
        "args": ["-t", "memory"],
        "requires_root": True,
        "description": "Retrieves physical RAM slots, installed DIMMs, speed, and maximum supported motherboard capacity.",
    },
    "get_gpu_info": {
        "binary": "lspci",
        "args": ["-nn"],
        "requires_root": False,
        "description": "Retrieves PCIe display/GPU controllers and vendor/device IDs.",
    },
}

SAFE_SCALAR_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
SHELL_METACHAR_RE = re.compile(r"[;&|`$><\n\r]")


class SecurityViolationError(ValueError):
    """Raised when an untrusted or unauthorized tool/argument is requested."""


def validate_scalar_arg(value: str) -> str:
    """Rejects any argument containing shell metacharacters or unsafe tokens."""
    if not isinstance(value, str) or SHELL_METACHAR_RE.search(value) or not SAFE_SCALAR_RE.fullmatch(value):
        raise SecurityViolationError(f"Rejected unsafe scalar argument: {value!r}")
    return value


def discover_available_tools() -> Dict[str, Dict[str, Any]]:
    """
    Distro portability check: uses shutil.which() at startup to build a dynamic
    registry of only the binaries actually installed on this Linux host.
    """
    available: Dict[str, Dict[str, Any]] = {}
    for tool_id, spec in STATIC_TOOL_SPECS.items():
        resolved_path = shutil.which(spec["binary"])
        available[tool_id] = {
            **spec,
            "resolved_path": resolved_path,
            "available": resolved_path is not None,
        }
    return available


def execute_allowlisted_command(
    tool_id: str,
    available_tools: Optional[Dict[str, Dict[str, Any]]] = None,
    use_sudo: bool = False,
    timeout: int = 5,
) -> Dict[str, Any]:
    """
    Executes an allow-listed binary directly via OS execve (shell=False).
    Never invokes /bin/sh.
    """
    if tool_id not in STATIC_TOOL_SPECS:
        raise SecurityViolationError(f"Unauthorized tool execution attempt: {tool_id!r}")

    registry = available_tools if available_tools is not None else discover_available_tools()
    tool_meta = registry.get(tool_id)
    if not tool_meta or not tool_meta.get("resolved_path"):
        return {
            "ok": False,
            "error": f"Tool binary '{STATIC_TOOL_SPECS[tool_id]['binary']}' is not installed on this system.",
            "stdout": "",
        }

    cmd: List[str] = [tool_meta["resolved_path"]] + list(tool_meta["args"])
    if use_sudo:
        sudo_path = shutil.which("sudo")
        if not sudo_path:
            return {"ok": False, "error": "sudo binary not found.", "stdout": ""}
        # -n (non-interactive) ensures CLI never hangs waiting for a password prompt
        cmd = [sudo_path, "-n"] + cmd

    try:
        result = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return {"ok": True, "stdout": result.stdout, "error": None}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Tool '{tool_id}' timed out after {timeout}s.", "stdout": ""}
    except subprocess.CalledProcessError as exc:
        err_msg = (exc.stderr or str(exc)).strip()
        return {"ok": False, "error": f"Command failed ({exc.returncode}): {err_msg}", "stdout": ""}
    except Exception as exc:
        return {"ok": False, "error": f"Execution failed: {exc}", "stdout": ""}
