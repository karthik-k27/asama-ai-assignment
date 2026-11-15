"""Security, Parser, Portability, Anti-Hallucination & Multi-Turn Tests for Problem 2."""

import unittest

from hw_assistant.llm_router import HardwareAssistant
from hw_assistant.parsers import (
    parse_dmidecode_memory,
    parse_lsblk_json,
    parse_lscpu,
    parse_lspci_gpu,
)
from hw_assistant.security import (
    SecurityViolationError,
    execute_allowlisted_command,
    validate_scalar_arg,
)


SAMPLE_DMIDECODE = """# dmidecode 3.5
Handle 0x001A, DMI type 16, 23 bytes
Physical Memory Array
\tMaximum Capacity: 64 GB
\tNumber Of Devices: 2

Handle 0x001B, DMI type 17, 92 bytes
Memory Device
\tSize: 16 GB
\tLocator: DIMM A1
\tType: DDR5
\tSpeed: 4800 MT/s

Handle 0x001C, DMI type 17, 92 bytes
Memory Device
\tSize: No Module Installed
\tLocator: DIMM B1
"""

SAMPLE_LSBLK = """{
  "blockdevices": [
    {"name": "nvme0n1", "size": "512G", "type": "disk", "mountpoint": null, "model": "Samsung SSD 980", "rota": false, "tran": "nvme"}
  ]
}"""

SAMPLE_LSPCI = """00:02.0 VGA compatible controller [0300]: Intel Corporation Alder Lake-P Integrated Graphics [8086:46a6] (rev 0c)
00:14.0 USB controller [0c03]: Intel Corporation Alder Lake PCH USB 3.2 xHCI Host Controller [8086:51ed]
"""


class TestLinuxHardwareAssistant(unittest.TestCase):
    def test_allowlist_blocks_arbitrary_and_injection_commands(self) -> None:
        with self.assertRaises(SecurityViolationError):
            execute_allowlisted_command("rm -rf /")

        with self.assertRaises(SecurityViolationError):
            execute_allowlisted_command("run_command('dmidecode; rm -rf /')")

        with self.assertRaises(SecurityViolationError):
            validate_scalar_arg("nvme0n1; cat /etc/shadow")

        self.assertEqual(validate_scalar_arg("nvme0n1"), "nvme0n1")

    def test_parsers_extract_structured_data(self) -> None:
        dmi = parse_dmidecode_memory(SAMPLE_DMIDECODE)
        self.assertTrue(dmi["dmi_available"])
        self.assertEqual(dmi["maximum_capacity"], "64 GB")
        self.assertEqual(dmi["total_slots"], 2)
        self.assertEqual(dmi["used_slots"], 1)
        self.assertEqual(dmi["free_slots"], 1)

        blk = parse_lsblk_json(SAMPLE_LSBLK)
        self.assertEqual(blk["disk_count"], 1)
        self.assertTrue(blk["disks"][0]["is_nvme"])

        gpu = parse_lspci_gpu(SAMPLE_LSPCI)
        self.assertEqual(gpu["gpu_count"], 1)
        self.assertIn("Intel Corporation", gpu["gpus"][0])

    def test_multi_turn_followup_and_cache_hit(self) -> None:
        calls = []

        def fake_runner(tool_name: str):
            calls.append(tool_name)
            if tool_name == "get_memory_topology":
                return {
                    "available": True,
                    "total_gb": 16.0,
                    "dmi_available": True,
                    "maximum_capacity": "64 GB",
                    "total_slots": 2,
                    "used_slots": 1,
                    "free_slots": 1,
                    "installed_modules": [{"size": "16 GB", "type": "DDR5", "speed": "4800 MT/s", "locator": "DIMM A1"}],
                }
            if tool_name == "get_gpu_info":
                return {"available": True, **parse_lspci_gpu(SAMPLE_LSPCI)}
            return {"available": False}

        assistant = HardwareAssistant(tool_runner=fake_runner)

        # Turn 1: RAM question
        r1 = assistant.ask("Can I upgrade my RAM, and by how much?")
        self.assertIn("16.0 GB", r1["answer"])
        self.assertIn("64 GB", r1["answer"])
        self.assertEqual(r1["tools_executed"], ["get_memory_topology"])

        # Turn 2: Follow-up GPU question
        r2 = assistant.ask("and what about the GPU?")
        self.assertIn("Intel Corporation", r2["answer"])
        self.assertEqual(r2["tools_executed"], ["get_gpu_info"])

        # Turn 3: Repeated RAM question hits cache without re-running tool
        r3 = assistant.ask("How many RAM slots did I have?")
        self.assertEqual(r3["tools_executed"], [])
        self.assertEqual(r3["cache_hits"], ["get_memory_topology"])

    def test_anti_hallucination_when_root_unavailable(self) -> None:
        def unprivileged_runner(tool_name: str):
            if tool_name == "get_memory_topology":
                return {"available": True, "total_gb": 16.0, "available_gb": 10.5, "dmi_available": False}
            return {"available": False}

        assistant = HardwareAssistant(tool_runner=unprivileged_runner)
        res = assistant.ask("How many RAM slots do I have?")
        self.assertIn("cannot be determined", res["answer"])
        self.assertIn("dmidecode", res["answer"])


if __name__ == "__main__":
    unittest.main()
