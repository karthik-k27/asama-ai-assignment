"""Deterministic Output Parsers & Normalizers for Linux Hardware Utilities."""

import json
import re
from typing import Any, Dict, List, Optional


def parse_lscpu(stdout: str) -> Dict[str, Any]:
    raw: Dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            raw[key.strip()] = val.strip()

    return {
        "architecture": raw.get("Architecture", "Data unavailable"),
        "model_name": raw.get("Model name", "Data unavailable"),
        "cpus": int(raw["CPU(s)"]) if raw.get("CPU(s)", "").isdigit() else None,
        "threads_per_core": int(raw["Thread(s) per core"]) if raw.get("Thread(s) per core", "").isdigit() else None,
        "cores_per_socket": int(raw["Core(s) per socket"]) if raw.get("Core(s) per socket", "").isdigit() else None,
        "virtualization": raw.get("Virtualization", "Data unavailable"),
    }


def parse_free_bytes(stdout: str) -> Dict[str, Any]:
    for line in stdout.splitlines():
        if line.lower().startswith("mem:"):
            parts = line.split()
            if len(parts) >= 4:
                total_b = int(parts[1])
                used_b = int(parts[2])
                free_b = int(parts[3])
                avail_b = int(parts[6]) if len(parts) >= 7 else free_b
                return {
                    "total_bytes": total_b,
                    "total_gb": round(total_b / (1024 ** 3), 2),
                    "used_gb": round(used_b / (1024 ** 3), 2),
                    "available_gb": round(avail_b / (1024 ** 3), 2),
                }
    return {"total_bytes": None, "total_gb": None, "used_gb": None, "available_gb": None}


def parse_proc_meminfo(content: str) -> Dict[str, Any]:
    mem_total_kb: Optional[int] = None
    mem_avail_kb: Optional[int] = None
    for line in content.splitlines():
        if line.startswith("MemTotal:"):
            m = re.search(r"(\d+)", line)
            if m:
                mem_total_kb = int(m.group(1))
        elif line.startswith("MemAvailable:"):
            m = re.search(r"(\d+)", line)
            if m:
                mem_avail_kb = int(m.group(1))

    return {
        "total_gb": round(mem_total_kb / (1024 ** 2), 2) if mem_total_kb is not None else None,
        "available_gb": round(mem_avail_kb / (1024 ** 2), 2) if mem_avail_kb is not None else None,
        "source": "/proc/meminfo",
    }


def parse_dmidecode_memory(stdout: str) -> Dict[str, Any]:
    max_cap_match = re.search(r"Maximum Capacity:\s*(.+)", stdout)
    num_devices_match = re.search(r"Number Of Devices:\s*(\d+)", stdout)

    # Split into Memory Device blocks
    blocks = re.split(r"Memory Device\s*\n", stdout)[1:]
    installed_modules: List[Dict[str, str]] = []
    empty_slots = 0

    for blk in blocks:
        size_m = re.search(r"^\s*Size:\s*(.+)$", blk, re.MULTILINE)
        if not size_m:
            continue
        size_val = size_m.group(1).strip()
        if "No Module Installed" in size_val:
            empty_slots += 1
            continue

        loc_m = re.search(r"^\s*Locator:\s*(.+)$", blk, re.MULTILINE)
        type_m = re.search(r"^\s*Type:\s*(.+)$", blk, re.MULTILINE)
        speed_m = re.search(r"^\s*Speed:\s*(.+)$", blk, re.MULTILINE)
        installed_modules.append(
            {
                "locator": loc_m.group(1).strip() if loc_m else "Unknown",
                "size": size_val,
                "type": type_m.group(1).strip() if type_m else "Unknown",
                "speed": speed_m.group(1).strip() if speed_m else "Unknown",
            }
        )

    total_slots = (
        int(num_devices_match.group(1))
        if num_devices_match
        else (len(installed_modules) + empty_slots)
    )
    return {
        "dmi_available": True,
        "maximum_capacity": max_cap_match.group(1).strip() if max_cap_match else "Data unavailable",
        "total_slots": total_slots,
        "used_slots": len(installed_modules),
        "free_slots": max(0, total_slots - len(installed_modules)),
        "installed_modules": installed_modules,
    }


def parse_lsblk_json(stdout: str) -> Dict[str, Any]:
    data = json.loads(stdout)
    disks: List[Dict[str, Any]] = []
    for dev in data.get("blockdevices", []):
        if dev.get("type") == "disk":
            tran = (dev.get("tran") or "").lower()
            name = dev.get("name") or ""
            is_nvme = tran == "nvme" or name.startswith("nvme")
            disks.append(
                {
                    "name": name,
                    "size": dev.get("size", "Unknown"),
                    "model": (dev.get("model") or "Unknown").strip(),
                    "transport": tran or ("nvme" if is_nvme else "unknown"),
                    "is_nvme": is_nvme,
                    "rotational": bool(dev.get("rota")),
                }
            )
    return {"disks": disks, "disk_count": len(disks)}


def parse_lspci_gpu(stdout: str) -> Dict[str, Any]:
    gpus: List[str] = []
    for line in stdout.splitlines():
        if any(k in line for k in ("VGA compatible controller", "3D controller", "Display controller")):
            gpus.append(line.strip())
    return {
        "gpus": gpus,
        "gpu_count": len(gpus),
    }
