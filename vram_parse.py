#!/usr/bin/env python3
"""VRAM output parsing for oom_guard.sh.

The guard's rocm-smi/nvidia-smi parsing used to live inline in the bash
script as python heredocs. It lives here now so teh exact code the guard
runs is directly testable against real captured tool output (see
tests/fixtures/) and the bash side stays a thin call.

Format facts this parser is built on (checked against AMD's rocm_smi_lib
source, python_smi_tools/rocm_smi.py, showMemInfo/printLog):
  - rocm-smi --showmeminfo vram prints per-GPU "VRAM Total Memory (B)" and
    "VRAM Total Used Memory (B)" lines. There is NO "VRAM Free Memory (B)"
    field, free has to be computed as total - used. an earlier version of
    this parser looked for a "Free" field that the real tool has never
    emitted in any release, which made VRAM monitoring a silent no-op on
    real hardware; a stubbed test had hidden that.
  - with --json, the same lines land in {"card0": {"<field>": "<value>"}}
    with STRING values (rocm-smi builds the JSON by splitting its own text
    output).

CLI: vram_parse.py <format> <payload>
  formats: rocm-json, rocm-text
Prints one "<index> <free_mb>" line per GPU, same bash contract as before:
empty output means "unparseable / no rows".
"""

import json
import re
import sys


def _to_int(value):
    """rocm-smi JSON carries the values as strings (it builds them from its
    own text output), plain text values are just digits. accept both."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _free_mb(total_b, used_b):
    if total_b is None or used_b is None or total_b <= 0 or used_b < 0:
        return None
    return (total_b - used_b) // 1024 // 1024


def parse_rocm_json(payload: str):
    """rocm-smi --showmeminfo vram --json:
    {"card0": {"VRAM Total Memory (B)": "...", "VRAM Total Used Memory (B)": "..."}, ...}"""
    try:
        data = json.loads(payload)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    rows = []
    for card, info in data.items():
        if not isinstance(info, dict):
            continue
        idx = int(str(card).replace("card", "")) if str(card).startswith("card") else 0
        total_b = _to_int(info.get("VRAM Total Memory (B)"))
        used_b = _to_int(info.get("VRAM Total Used Memory (B)"))
        free_mb = _free_mb(total_b, used_b)
        if free_mb is not None and free_mb > 0:
            rows.append((idx, free_mb))
    return rows


def parse_rocm_text(payload: str):
    """rocm-smi --showmeminfo vram (text): every line repeats the GPU[N]
    header so a Total and its Used line are seperate lines, group by GPU
    index across lines then compute free per GPU."""
    totals = {}
    used = {}
    for m in re.finditer(r"GPU\[(\d+)\]\s*:\s*VRAM Total Memory \(B\):\s*(\d+)", payload):
        totals[int(m.group(1))] = int(m.group(2))
    for m in re.finditer(r"GPU\[(\d+)\]\s*:\s*VRAM Total Used Memory \(B\):\s*(\d+)", payload):
        used[int(m.group(1))] = int(m.group(2))
    rows = []
    for idx in sorted(totals):
        free_mb = _free_mb(totals[idx], used.get(idx))
        if free_mb is not None and free_mb > 0:
            rows.append((idx, free_mb))
    return rows



PARSERS = {
    "rocm-json": parse_rocm_json,
    "rocm-text": parse_rocm_text,
}


def main(argv):
    if len(argv) != 3 or argv[1] not in PARSERS:
        print(f"usage: vram_parse.py {{{'|'.join(PARSERS)}}} <payload>", file=sys.stderr)
        return 2
    # payload goes over argv not stdin: oom_guard.sh calls this from bash
    # where a heredoc on the same command silently wins over a pipe and
    # hands the parser an empty stdin. a real bug this repo shipped once.
    for idx, free_mb in PARSERS[argv[1]](argv[2]):
        print(idx, free_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
