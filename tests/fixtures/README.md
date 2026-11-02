# Test fixtures

Real captured outputs, not fabricated test data. Provenance:

- `meminfo_healthy.txt` -- a real `/proc/meminfo` captured from a live
  Linux kernel (Docker VM, 12GB RAM class machine) at rest.
- `meminfo_pressure.txt` -- a real `/proc/meminfo` captured from the same
  kernel while a real allocation loop had genuinely driven `MemAvailable`
  down to ~159MB. The kernel computed the value; nothing was edited.
- `rocm_smi_vram_text.txt` -- real `rocm-smi --showmeminfo vram` output
  pasted by a user with an actual AMD Instinct MI300 (96GB) in
  ollama/ollama#17271. Values match the hardware spec (96GiB total).
- `rocm_smi_vram.json` -- the `--json` form of the same real capture.
  rocm-smi builds its JSON by splitting its own text lines, so the field
  values arrive as strings (see rocm_smi_lib's `printLog`/`JSON_DATA`
  handling in python_smi_tools/rocm_smi.py). Constructed mechanically
  from the text capture with the same real values, string-typed exactly
  as the tool emits them.

The guard is driven against these through its real override hooks
(`OOM_GUARD_MEMINFO`, `OOM_GUARD_ROCM_SMI`), which are the same mechanism
documented in oom_guard.sh for lab/testing use on machines without the
corresponding hardware.
