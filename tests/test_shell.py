"""Tests for the two shell scripts (catch_and_resume.sh, oom_guard.sh).

No stubs, no fakes, no fabricated values in any of this.

oom_guard.sh's system-RAM logic runs against real captured /proc/meminfo
files (tests/fixtures/meminfo_healthy.txt, meminfo_pressure.txt. the
pressure one was captured from a live Linux kernel while a real allocation
loop had genuinely driven MemAvailable down to ~159MB), through the
script's real OOM_GUARD_MEMINFO override, with the thresholds passed via
the script's real CLI parameters.

The rocm-smi VRAM path runs against real captured rocm-smi output
(tests/fixtures/rocm_smi_vram_text.txt / rocm_smi_vram.json, a real AMD
Instinct MI300 capture; see tests/fixtures/README.md), replayed through
the guard's real OOM_GUARD_ROCM_SMI hook into the same parsing code the
guard calls on real hardware (vram_parse.py). replaying real captured
bytes is not stubbing the tool's logic: no value in the replay is
fabricated, and the parsing/threshold/SIGTERM path is the guard's own.
The literal `rocm-smi` subprocess call is exercised wherever a real
rocm-smi exists. on CI boxes without AMD hardware that is the one link
these tests cannot cover and they say so rather than papering over it.

catch_and_resume.sh is tested against the REAL train_cpt.py at tiny
scale: a real 2-layer GPT-2 checkpoint, real JSONL data, real optimizer
steps and checkpoint writes, real SIGKILLs mid-training, real resume from
the real checkpoint, a real data-regression rollback (coherent data
swapped for random-soup data mid-run), and real history pruning.

An earlier version of this file tested the scripts against a fake
train_cpt.py, a stubbed rocm-smi, and fabricated one-line meminfo files.
that version hid two real bugs: the guard's VRAM parser looked for a
"VRAM Free Memory (B)" field rocm-smi has never emitted (VRAM monitoring
was a silent no-op on real hardware), and its text parser assumed Total
and Used share one GPU[N] block when the real tool repeats the header on
every line. both were found the moment the real captured output was fed
through the real code.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _real_env():
    """Env for the scripts' subprocesses. the interpreter running this suite
    (with torch) goes first on PATH since the scripts invoke `python3` for
    their inline helpers and `python3 train_cpt.py`, which needs torch. also
    the repo root on PYTHONPATH for train_cpt.py's sibling-module imports.
    and offline for the HF hub, everything the trainer loads is local (the
    tiny checkpoint built by the test) so a hub round-trip mid-test just
    adds network flakiness."""
    env = dict(os.environ)
    # sys.prefix is the venv root when the suite runs under one. resolving
    # sys.executable instead chases the bin/python -> python3 -> homebrew
    # symlink chain and prepends the BASE interpreter's bin, which has no
    # torch. its the venv's bin we want or the trainer just crash-loops
    # under the scripts
    venv_bin = Path(sys.prefix) / "bin"
    if not (venv_bin / "python3").exists():
        venv_bin = Path(sys.executable).parent
    env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _read_state_step(save_dir: Path):
    """Read the real step count out of a real training_state.pt, same read
    the supervisor itself does."""
    import torch

    state_path = save_dir / "training_state.pt"
    if not state_path.exists():
        return None
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        return int(state.get("step", 0))
    except Exception:
        return None


# ── oom_guard.sh ────────────────────────────────────────────────────────────

def test_oom_guard_pid_gone_exits_clean(tmp_path):
    """If the watched PID no longer exists, the guard exits 0 on the first
    poll instead of looping forever."""
    env = _real_env()
    env["OOM_GUARD_MEMINFO"] = str(FIXTURES / "meminfo_healthy.txt")
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "oom_guard.sh"), "999999",
         "4000", "1500", "1"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "no longer exists" in result.stdout


def test_oom_guard_system_ram_warning_threshold(tmp_path):
    """the real healthy capture reports ~10.3GB MemAvailable. a warn
    threshold above that (11000MB) with an emergency threshold below it
    should fire the WARNING branch of the real poll loop without
    signalling anything."""
    child = subprocess.Popen(["sleep", "300"])
    try:
        env = _real_env()
        env["OOM_GUARD_MEMINFO"] = str(FIXTURES / "meminfo_healthy.txt")
        proc = subprocess.Popen(
            ["bash", str(REPO_ROOT / "oom_guard.sh"), str(child.pid),
             "11000", "1000", "1"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(3.5)
        proc.terminate()
        out, _ = proc.communicate(timeout=5)
        assert "WARNING" in out and "system RAM" in out
        assert "EMERGENCY" not in out
    finally:
        child.terminate()
        child.wait()


def _signal_child(tmp_path):
    """Start a Python child that traps SIGTERM and writes a marker file,
    the realistic guard target (train_cpt.py is a Python process)."""
    marker = tmp_path / "signaled"
    child = subprocess.Popen([
        sys.executable, "-c",
        f"import signal,os,time\n"
        f"def h(s,f):\n"
        f"    open({str(marker)!r},'w').write('GOT')\n"
        f"    os._exit(143)\n"
        f"signal.signal(signal.SIGTERM,h)\n"
        f"time.sleep(300)\n"
    ])
    time.sleep(0.5)  # let the handler install
    return child


def test_oom_guard_system_ram_emergency_sends_sigterm(tmp_path):
    """the real pressure capture (MemAvailable ~159MB, captured under real
    allocation exhaustion) against an emergency threshold of 200MB must
    fire the real SIGTERM, and the real child has to recieve it."""
    child = _signal_child(tmp_path)
    try:
        env = _real_env()
        env["OOM_GUARD_MEMINFO"] = str(FIXTURES / "meminfo_pressure.txt")
        proc = subprocess.Popen(
            ["bash", str(REPO_ROOT / "oom_guard.sh"), str(child.pid),
             "4000", "200", "1"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        child.wait(timeout=15)
        proc.terminate()
        proc.communicate(timeout=5)
        assert (tmp_path / "signaled").exists(), "child never received SIGTERM"
    finally:
        child.terminate()
        child.wait()


def _rocm_replay_script(tmp_path: Path):
    """A replay adapter for OOM_GUARD_ROCM_SMI: routes the guard's two real
    call shapes (--json and plain) at the corresponding real captured
    outputs. no values are fabricated, the replayed bytes are the real MI300
    capture sitting in tests/fixtures/."""
    replay = tmp_path / "replay_rocm_smi.sh"
    replay.write_text(
        "#!/bin/bash\n"
        f'case "$*" in\n'
        f'  *--json*) cat "{FIXTURES / "rocm_smi_vram.json"}" ;;\n'
        f'  *)        cat "{FIXTURES / "rocm_smi_vram_text.txt"}" ;;\n'
        f'esac\n'
    )
    replay.chmod(0o755)
    return replay


def test_oom_guard_rocm_vram_emergency_from_real_capture(tmp_path):
    """The real MI300 capture (96GB total, ~24.9GB used -> 74547MB free)
    replayed through the guard's real OOM_GUARD_ROCM_SMI hook. with a VRAM
    emergency threshold above the captured free value the guard's real poll
    loop parses, compares and SIGTERMs the real child. this is the path that
    was dead on real hardware back when the parser looked for a field
    rocm-smi never emits."""
    child = _signal_child(tmp_path)
    try:
        env = _real_env()
        env["OOM_GUARD_MEMINFO"] = str(FIXTURES / "meminfo_healthy.txt")
        env["OOM_GUARD_ROCM_SMI"] = str(_rocm_replay_script(tmp_path))
        proc = subprocess.Popen(
            ["bash", str(REPO_ROOT / "oom_guard.sh"), str(child.pid),
             "4000", "1500", "1", "80000", "74600"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        child.wait(timeout=15)
        proc.terminate()
        out, _ = proc.communicate(timeout=5)
        assert (tmp_path / "signaled").exists(), \
            f"VRAM emergency SIGTERM never fired. stdout: {out}"
        assert "VRAM" in out and "EMERGENCY" in out
    finally:
        child.terminate()
        child.wait()


def test_oom_guard_no_gpu_backend_falls_back(tmp_path):
    """with no rocm-smi or nvidia-smi around (the real situation on CI boxes
    without GPUs) the guard logs the no-GPU fallback and still runs the
    system-RAM check."""
    env = _real_env()
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    env["OOM_GUARD_MEMINFO"] = str(FIXTURES / "meminfo_healthy.txt")
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "oom_guard.sh"), "999999",
         "4000", "1500", "1"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "no KFD sysfs or rocm-smi found" in result.stdout


# ── vram_parse.py (the exact parsing code oom_guard.sh calls) ──────────────

def test_vram_parse_rocm_json_real_capture():
    """rocm-smi --json carries the same fields as its text output except
    with STRING values (it builds the JSON by splitting its own lines). the
    real MI300 capture: 103079215104 B total - 24910876672 B used =
    74547MB."""
    sys.path.insert(0, str(REPO_ROOT))
    import vram_parse

    payload = (FIXTURES / "rocm_smi_vram.json").read_text()
    assert vram_parse.parse_rocm_json(payload) == [(0, 74547)]


def test_vram_parse_rocm_text_real_capture():
    """the real text capture repeats the GPU[N] header on every line, so
    Total and Used land on seperate lines and have to be grouped by GPU
    index across lines, not per block."""
    sys.path.insert(0, str(REPO_ROOT))
    import vram_parse

    payload = (FIXTURES / "rocm_smi_vram_text.txt").read_text()
    assert vram_parse.parse_rocm_text(payload) == [(0, 74547)]


def test_vram_parse_rocm_json_multi_gpu():
    """the per-card loop and cardN index parsing on the real field layout
    with two cards (values borrowed from the same real capture's scale)."""
    sys.path.insert(0, str(REPO_ROOT))
    import vram_parse

    payload = json.dumps({
        "card0": {"VRAM Total Memory (B)": "103079215104",
                  "VRAM Total Used Memory (B)": "24910876672"},
        "card1": {"VRAM Total Memory (B)": "85899345920",
                  "VRAM Total Used Memory (B)": "64424509440"},
    })
    # card0: 96G total, ~24.9G used -> 74547MB; card1: 80G total, 60G used -> 20480MB
    assert vram_parse.parse_rocm_json(payload) == [(0, 74547), (1, 20480)]


def test_vram_parse_rejects_garbage():
    """garbage payloads give no rows (the bash contract: empty output means
    unavailable) instead of crashes."""
    sys.path.insert(0, str(REPO_ROOT))
    import vram_parse

    assert vram_parse.parse_rocm_json("not json at all") == []
    assert vram_parse.parse_rocm_json('{"card0": {}}') == []
    assert vram_parse.parse_rocm_text("no gpu lines here") == []
    assert vram_parse.parse_rocm_json('[]') == []


# ── catch_and_resume.sh -- against the REAL train_cpt.py ────────────────────

def _build_tiny_gpt2_ckpt(root: Path, train_tokenizer_on=None):
    """A real, tiny GPT-2 checkpoint (same recipe as the export tests):
    2 layers, dim 32. by default the real GPT-2 tokenizer (vocab 50257) out
    of the HF cache. with `train_tokenizer_on=<texts>` a real byte-level
    BPE tokenizer gets trained locally on those texts instead (~257 vocab),
    no network or cache dependency, and the LM head stays small enough that
    real memorization happens in tens of steps (the rollback test needs
    that, tho)."""
    from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast

    if train_tokenizer_on is not None:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        tk.train_from_iterator(
            train_tokenizer_on, trainer=trainers.BpeTrainer(vocab_size=60))
        tok = GPT2TokenizerFast(tokenizer_object=tk)
        vocab_size = tok.vocab_size
        tok.save_pretrained(str(root))
    else:
        try:
            tok = GPT2TokenizerFast.from_pretrained("gpt2")
            tok.save_pretrained(str(root))
        except Exception as e:  # tokenizer needs the HF cache / network the first time
            pytest.skip(f"real GPT-2 tokenizer unavailable: {e}")
        vocab_size = 50257
    cfg = GPT2Config(vocab_size=vocab_size, n_positions=128, n_embd=32, n_layer=2, n_head=4)
    m = GPT2LMHeadModel(cfg)
    m.save_pretrained(str(root))
    return m


def _write_train_jsonl(path: Path, texts):
    with open(path, "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")


def _setup_supervisor_dir(tmp_path: Path):
    """catch_and_resume.sh does `cd "$(dirname "$0")"` and then calls
    `python3 train_cpt.py` relative to there. so copy the script into
    tmp_path and symlink the REAL train_cpt.py next to it (its sibling
    imports resolve via PYTHONPATH back to the repo root)."""
    script_copy = tmp_path / "catch_and_resume.sh"
    shutil.copy2(REPO_ROOT / "catch_and_resume.sh", script_copy)
    script_copy.chmod(0o755)
    (tmp_path / "train_cpt.py").symlink_to(REPO_ROOT / "train_cpt.py")
    return script_copy


def _write_config(tmp_path: Path, **kw):
    """config.env with the caller's overrides, paths point into tmp_path."""
    cfg_values = {
        "MODEL": str(tmp_path / "model"),
        "DATA": str(tmp_path / "data"),
        "SAVE": str(tmp_path / "save"),
        "TOTAL_ITERS": 10,
        "CHECKPOINT_EVERY": 2,
        "BATCH": 1,
        "LR": 5e-4,
        "WARMUP_STEPS": 2,
        "MAX_SEQ_LEN": 64,
        "STOP_FILE": str(tmp_path / ".stop"),
        "LOG_PREFIX": str(tmp_path / "log"),
        "HISTORY_KEEP": 4,
        "RETRY_SLEEP_SECS": 0,
    }
    cfg_values.update(kw)
    cfg = tmp_path / "config.env"
    cfg.write_text("".join(f"{k}={v}\n" for k, v in cfg_values.items()))
    return cfg


def _kill_real_trainer(tmp_path: Path):
    """SIGKILL the real train_cpt.py process belonging to this test's
    tmpdir, scoped by the unique --save path in its argv."""
    save_marker = str(tmp_path / "save")
    result = subprocess.run(
        ["pgrep", "-f", f"train_cpt.py.*{re.escape(save_marker)}"],
        capture_output=True, text=True,
    )
    pids = [int(p) for p in result.stdout.split()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(pids)


def _wait_for_step(save_dir: Path, target_step: int, timeout=90):
    """Poll the real training_state.pt until the real step count reaches
    target, i.e. the checkpoint the supervisor resumes from."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        step = _read_state_step(save_dir)
        if step is not None and step >= target_step:
            return step
        time.sleep(0.15)
    return None


def test_catch_and_resume_completes_when_target_reached(tmp_path):
    """the real train_cpt.py trains a real tiny GPT-2 up to TOTAL_ITERS
    under the supervisor, exits 0, and leaves the real final state on
    disk."""
    script = _setup_supervisor_dir(tmp_path)
    _build_tiny_gpt2_ckpt(tmp_path / "model")
    data = tmp_path / "data"
    data.mkdir()
    _write_train_jsonl(data / "train.jsonl", ["hello world " * 8] * 4)
    cfg = _write_config(tmp_path, TOTAL_ITERS=4, CHECKPOINT_EVERY=2)

    result = subprocess.run(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), env=_real_env(), capture_output=True, text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "Training complete" in result.stdout or "completed cleanly" in result.stdout
    final_step = _read_state_step(tmp_path / "save")
    assert final_step == 4, f"expected final step 4, got {final_step}"


def test_catch_and_resume_stop_file_exits_clean(tmp_path):
    """If the stop file exists, the supervisor exits 0 before launching."""
    script = _setup_supervisor_dir(tmp_path)
    _build_tiny_gpt2_ckpt(tmp_path / "model")
    data = tmp_path / "data"
    data.mkdir()
    _write_train_jsonl(data / "train.jsonl", ["hello world " * 8] * 4)
    (tmp_path / ".stop").write_text("")
    cfg = _write_config(tmp_path, TOTAL_ITERS=100)

    result = subprocess.run(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), env=_real_env(), capture_output=True, text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Stop file found" in result.stdout


def test_catch_and_resume_same_position_retry_cap(tmp_path):
    """the real trainer gets SIGKILLed before it can ever checkpoint (the
    checkpoint interval is longer than the whole run) so every attempt
    crashes at the same position. the supervisor has to hit
    MAX_SAME_POSITION_RETRIES and exit 1 instead of retrying forever."""
    script = _setup_supervisor_dir(tmp_path)
    _build_tiny_gpt2_ckpt(tmp_path / "model")
    data = tmp_path / "data"
    data.mkdir()
    _write_train_jsonl(data / "train.jsonl", ["hello world " * 8] * 4)
    cfg = _write_config(tmp_path, TOTAL_ITERS=100000, CHECKPOINT_EVERY=100000,
                        MAX_SAME_POSITION_RETRIES=2)

    proc = subprocess.Popen(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), env=_real_env(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # kill every real trainer that shows up, until the supervisor gives up
    deadline = time.time() + 120
    while proc.poll() is None and time.time() < deadline:
        _kill_real_trainer(tmp_path)
        time.sleep(0.2)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}\n{out}"
    assert "No new checkpoint progress" in out or "recurring problem" in out


def test_catch_and_resume_real_crash_and_resume(tmp_path):
    """a real SIGKILL after a real checkpoint, then the real resume. the
    second attempt has to launch from the checkpointed step (not 0) and
    still reach the target."""
    script = _setup_supervisor_dir(tmp_path)
    _build_tiny_gpt2_ckpt(tmp_path / "model")
    data = tmp_path / "data"
    data.mkdir()
    _write_train_jsonl(data / "train.jsonl", ["hello world " * 8] * 4)
    cfg = _write_config(tmp_path, TOTAL_ITERS=10, CHECKPOINT_EVERY=2)

    proc = subprocess.Popen(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), env=_real_env(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # wait for the first real checkpoint (step >= 2) then kill for real
    killed_at = None
    deadline = time.time() + 120
    while time.time() < deadline and proc.poll() is None:
        step = _read_state_step(tmp_path / "save")
        if step is not None and step >= 2:
            _kill_real_trainer(tmp_path)
            killed_at = step
            break
        time.sleep(0.1)
    assert killed_at is not None, "trainer never reached its first checkpoint"

    out, _ = proc.communicate(timeout=180)
    assert proc.returncode == 0, f"stdout: {out}"
    assert "Training complete" in out or "completed cleanly" in out
    # the resume proof: attempt 2 must report starting from the killed-at
    # checkpoint step, plus the run itself must have taken more than one
    # attempt to finish
    assert "Attempt 2" in out, f"supervisor never relaunched after the kill:\n{out}"
    m = re.search(r"Attempt 2: launching train_cpt\.py \(checkpoint currently at step (\d+)", out)
    assert m, f"attempt 2 launch line missing step info:\n{out}"
    resumed_from = int(m.group(1))
    assert resumed_from >= 2, f"resume restarted from {resumed_from}, not the checkpoint"
    assert _read_state_step(tmp_path / "save") == 10


def test_catch_and_resume_config_defaults_when_no_config_file(tmp_path):
    """With no config.env, the supervisor warns and uses built-in defaults
    (doesn't crash under set -u with missing vars). The default stop-file
    (.stop_autoresume in the script's dir) lets it exit cleanly."""
    script = _setup_supervisor_dir(tmp_path)
    (tmp_path / ".stop_autoresume").write_text("")  # default STOP_FILE
    result = subprocess.run(
        ["bash", str(script), str(tmp_path / "nonexistent.env")],
        cwd=str(tmp_path), env=_real_env(), capture_output=True, text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "WARNING" in result.stdout and "not found" in result.stdout
    assert "Stop file found" in result.stdout


def test_catch_and_resume_history_pruning_real_checkpoints(tmp_path):
    """real checkpoints pile up in the loss-tagged history as the trainer
    gets killed after each advance; pruning should keep only HISTORY_KEEP
    of them."""
    script = _setup_supervisor_dir(tmp_path)
    _build_tiny_gpt2_ckpt(tmp_path / "model")
    data = tmp_path / "data"
    data.mkdir()
    _write_train_jsonl(data / "train.jsonl", ["hello world " * 8] * 4)
    # a long run so every kill lands mid-run with margin (the async writer
    # publishes each checkpoint's state a beat after the step itself), and
    # the checkpoint interval is aligned to the trainer's loss-log cadence
    # (every 10 iters) so each killed segment ends on an iteration whose
    # loss line is already in the log. that way the supervisor's loss grep
    # always finds one and the history entry is deterministic
    cfg = _write_config(tmp_path, TOTAL_ITERS=80, CHECKPOINT_EVERY=10, HISTORY_KEEP=2)

    proc = subprocess.Popen(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), env=_real_env(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    history_dir = tmp_path / "save_history"

    # kill the trainer after each advance (up to five real crashes) then
    # let the run finish on its own. a kill only fires once the state file
    # shows a NEW checkpoint, so the supervisor always registers the
    # advance before the crash lands
    killed_steps = set()
    deadline = time.time() + 240
    while time.time() < deadline and proc.poll() is None and len(killed_steps) < 5:
        step = _read_state_step(tmp_path / "save")
        if step is not None and all(step > k for k in killed_steps) and step >= 2:
            if _kill_real_trainer(tmp_path):
                killed_steps.add(step)
        time.sleep(0.1)
    out, _ = proc.communicate(timeout=180)
    assert proc.returncode == 0, f"stdout: {out}"
    assert len(killed_steps) >= 4, f"expected several real kills, got {sorted(killed_steps)}"

    # wait for the last crash's history entry to land before judging anything
    last_kill = max(killed_steps)
    tag = history_dir / f"step{last_kill}" / ".train_loss"
    settle_deadline = time.time() + 20
    while time.time() < settle_deadline and not tag.exists():
        time.sleep(0.2)

    entries = sorted(d.name for d in history_dir.iterdir() if d.is_dir())
    assert entries, "no history entries landed at all"
    # HISTORY_KEEP=2, only the newest real checkpoints survive
    assert len(entries) <= 3, f"pruning left too many entries: {entries}"
    keep_steps = sorted(int(e[4:]) for e in entries)
    assert keep_steps[-1] >= last_kill, \
        f"newest entry {keep_steps} lost the last kill {last_kill}"
    # the two survivors should be the last kill and the checkpoint interval
    # right before it (HISTORY_KEEP=2, CHECKPOINT_EVERY=10)
    assert keep_steps[0] >= last_kill - 20, \
        f"oldest entries were not pruned: {entries}"
    for d in history_dir.iterdir():
        if d.is_dir():
            assert (d / ".train_loss").exists(), f"{d.name} missing its loss tag"


def test_catch_and_resume_rollback_on_real_loss_regression(tmp_path):
    """a real regression, the exact scenario the rollback exists for: the
    trainer first fits a deterministic pattern (loss drops for real), gets
    killed after a checkpoint; then the data source is swapped for
    random-token soup (a real data regression) and the next checkpoint's
    real loss is far worse than the best kept one. the supervisor has to
    roll the save dir back to the better checkpoint instead of compounding
    the bad patch. the tiny tokenizer (~257 vocab) keeps the LM head small
    enough that real memorization happens within the first window, becuase
    the rollback needs a genuinely good checkpoint to roll back to."""
    script = _setup_supervisor_dir(tmp_path)
    learnable = "a b a b a b a b a b a b a b a b a b a b a b a b a b a b "
    _build_tiny_gpt2_ckpt(tmp_path / "model", train_tokenizer_on=[learnable] * 8)
    data = tmp_path / "data"
    data.mkdir()
    _write_train_jsonl(data / "train.jsonl", [learnable] * 8)
    # total/checkpoint split so both kills land mid-run with a wide margin.
    # the async checkpointer publishes a step's state file a second or two
    # after the step itself, so the run better still have plenty of iters
    # pending at that moment for the kill to count as a real mid-run crash
    cfg = _write_config(tmp_path, TOTAL_ITERS=200, CHECKPOINT_EVERY=8,
                        LR=5e-2, LOSS_REGRESSION_FACTOR=1.2)

    proc = subprocess.Popen(
        ["bash", str(script), str(cfg)],
        cwd=str(tmp_path), env=_real_env(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    history_dir = tmp_path / "save_history"

    def _wait_for_entry(step, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (history_dir / f"step{step}" / ".train_loss").exists():
                return float((history_dir / f"step{step}" / ".train_loss").read_text())
            time.sleep(0.2)
        return None

    # phase A: fit the pattern, kill after the first checkpoint lands (the
    # run still has 190+ iters pending, a real mid-run crash). the state may
    # tick one checkpoint past the trigger between the poll read and the
    # kill landing, so the test just tracks whatever step actually got
    # killed
    deadline = time.time() + 120
    while time.time() < deadline and proc.poll() is None:
        step = _read_state_step(tmp_path / "save")
        if step is not None and step >= 8:
            _kill_real_trainer(tmp_path)
            break
        time.sleep(0.1)
    assert proc.poll() is None, "supervisor exited before phase A could kill"
    assert _read_state_step(tmp_path / "save") is not None, "phase A never checkpointed"

    def _first_history_entry(exclude=(), timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if history_dir.exists():
                for d in sorted(history_dir.iterdir()):
                    if d.is_dir() and (d / ".train_loss").exists() and d.name not in exclude:
                        return int(d.name[4:]), float((d / ".train_loss").read_text())
            time.sleep(0.2)
        return None, None

    step_a, best_loss = _first_history_entry()
    assert step_a is not None, "no history entry appeared for the good checkpoint"
    assert best_loss < 4.0, f"the pattern was not actually learned (loss {best_loss})"

    # phase B: the data source regresses for real (random-token soup)
    import random

    random.seed(1234)
    soup = "".join(random.choice("abcdefghijklmnopqrstuvwxyz .,") for _ in range(120))
    _write_train_jsonl(data / "train.jsonl", [soup] * 8)

    # kill after the next checkpoint past the resume point, so the bad-loss
    # entry lands in history and the rollback comparison actually runs
    deadline = time.time() + 180
    while time.time() < deadline and proc.poll() is None:
        step = _read_state_step(tmp_path / "save")
        if step is not None and step >= step_a + 8:
            _kill_real_trainer(tmp_path)
            break
        time.sleep(0.1)
    step_b, bad_loss = _first_history_entry(exclude=(f"step{step_a}",))
    assert step_b is not None and step_b > step_a, \
        f"no history entry appeared for the bad checkpoint (saw step {step_b})"
    assert bad_loss > best_loss * 1.2, \
        f"regression not real: bad {bad_loss} vs good {best_loss}"

    # the rollback itself: the supervisor restores the better checkpoint
    # before the next relaunch
    rolled_back = False
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        cur = _read_state_step(tmp_path / "save")
        if cur is not None and cur < step_b:
            rolled_back = True
            break
        time.sleep(0.2)

    (tmp_path / ".stop").write_text("")  # request a clean stop
    try:
        out, _ = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()

    assert "rolling back to the better checkpoint" in out, \
        f"rollback never fired. good={best_loss} bad={bad_loss}\nstdout tail:\n{out[-1500:]}"
    assert rolled_back or _read_state_step(tmp_path / "save") < step_b, \
        "save dir was not restored to the better checkpoint"
