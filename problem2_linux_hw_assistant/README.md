# Problem 2: Local Linux Hardware Q&A Assistant (Asama.ai)

A privacy-first, deterministic Linux hardware inspection assistant with **strict allow-listed tool execution (`shell=False`)**, **scoped root fallback**, **structured JSON parsers**, and **multi-turn session caching**.

## Key Safety & Engineering Features
- **No Arbitrary Shell Execution**: The LLM never constructs bash commands. Only static tool IDs in `STATIC_TOOL_SPECS` can be executed via `subprocess.run(..., shell=False, timeout=5)` (direct `execve` syscall).
- **Privilege Escalation & Non-Root Fallback**: Attempts non-interactive `sudo -n dmidecode -t memory` when available; gracefully falls back to `/proc/meminfo` and `free -b` while explicitly stating missing root-only fields instead of hallucinating.
- **Distro Portability**: Discovers installed utilities at startup via `shutil.which()` and exposes only available tools in the dynamic schema registry.
- **Offline Privacy**: Runs locally with deterministic grounded synthesis or local **Ollama** (`ASAMA_USE_OLLAMA=1`).

## Quick Start
```bash
python3 -m unittest test_problem2.py -v
python3 -m hw_assistant.cli "How much RAM and CPU do I have?"
python3 -m hw_assistant.cli
```
