#!/usr/bin/env python3
"""Export a checkpoint to a single consolidated `model.safetensors` file.

Useful when a training run produces sharded checkpoints and you want a single
file for downstream tools that don't read `model.safetensors.index.json`.

Also copies every non-weight file from the source checkpoint (config.json,
tokenizer files, chat_template.jinja, generation_config.json, etc.) into the
output directory, matching mtp_head.py's copy behavior -- without this, the
output directory isn't loadable via `from_pretrained` on its own (it would
have weights but no config/tokenizer).
"""

import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="Source checkpoint directory.")
    ap.add_argument("--dst", help="Output directory.")
    ap.add_argument("--max-shard-bytes", type=int, default=None,
                    help="If set, shard output so no file exceeds this many bytes.")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU required).")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return

    if not args.src or not args.dst:
        ap.error("--src and --dst are required (unless --selftest).")

    from safetensors.torch import load_file, save_file

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    elif os.path.exists(os.path.join(args.src, "model.safetensors")):
        shard_files = ["model.safetensors"]
    else:
        raise SystemExit(f"No safetensors found in {args.src}")

    tensors = {}
    for shard in shard_files:
        tensors.update(load_file(os.path.join(args.src, shard)))

    os.makedirs(args.dst, exist_ok=True)

    # Copy every non-weight file from src into dst (config.json, tokenizer
    # files, chat_template.jinja, generation_config.json, etc.) -- same
    # approach as mtp_head.py: copy everything EXCEPT the weight shards and
    # the old index (those are rewritten below with the consolidated set).
    # Without this, dst has weights but no config/tokenizer and isn't
    # loadable via from_pretrained on its own.
    shard_set = set(shard_files)
    copied = 0
    for fname in os.listdir(args.src):
        if fname in shard_set or fname == "model.safetensors.index.json":
            continue
        s = os.path.join(args.src, fname)
        d = os.path.join(args.dst, fname)
        if os.path.isfile(s):
            shutil.copy2(s, d)
            copied += 1
    print(f"[export_safetensors] copied {copied} non-weight file(s) "
          f"(config/tokenizer/etc.) from {args.src} to {args.dst}")

    if args.max_shard_bytes:
        from expand_model import write_sharded
        write_sharded(tensors, args.dst, args.max_shard_bytes, log_prefix="export_safetensors")
    else:
        save_file(tensors, os.path.join(args.dst, "model.safetensors"))
        print(f"[export_safetensors] wrote {len(tensors)} tensors to {args.dst}/model.safetensors")


def _self_test():
    """Self-test: build a tiny real GPT-2 checkpoint, hand-split its
    model.safetensors into 2 shards + index.json (the layout HF sharding
    produces), consolidate it back via main(), then run main() again with
    --max-shard-bytes for the sharded-output path (no GPU)."""
    import sys
    import tempfile

    import torch
    from safetensors.torch import load_file, save_file

    print("[selftest] export_safetensors: consolidate + re-shard a tiny real checkpoint")
    with tempfile.TemporaryDirectory() as td:
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

        src = os.path.join(td, "src")
        # same tiny config as tests/test_all.py's _build_tiny_gpt2_checkpoint
        cfg = GPT2Config(vocab_size=50257, n_positions=128, n_embd=32,
                         n_layer=2, n_head=4)
        GPT2LMHeadModel(cfg).save_pretrained(src)
        try:
            GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(src)
        except Exception as exc:
            print(f"[selftest] note: GPT-2 tokenizer unavailable ({exc}) -- "
                  f"config.json still exercises the non-weight copy below")
        expected_non_weight = {f for f in os.listdir(src)
                               if not f.endswith(".safetensors")}

        # hand-split the single model.safetensors into 2 shards + index.json,
        # same layout save_pretrained(max_shard_size=...) produces
        single = os.path.join(src, "model.safetensors")
        tensors = load_file(single)
        keys = sorted(tensors.keys())
        half = len(keys) // 2
        shard_specs = [
            ("model-00001-of-00002.safetensors", keys[:half]),
            ("model-00002-of-00002.safetensors", keys[half:]),
        ]
        os.remove(single)
        weight_map = {}
        total_size = 0
        for fname, shard_keys in shard_specs:
            save_file({k: tensors[k] for k in shard_keys},
                      os.path.join(src, fname))
            total_size += os.path.getsize(os.path.join(src, fname))
            for k in shard_keys:
                weight_map[k] = fname
        with open(os.path.join(src, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size},
                       "weight_map": weight_map}, f)

        # consolidate via the real main() path, argv patched the same way the
        # pytest suite drives CLIs
        dst = os.path.join(td, "dst")
        old_argv = sys.argv
        sys.argv = ["export_safetensors.py", "--src", src, "--dst", dst]
        try:
            main()
        finally:
            sys.argv = old_argv

        consolidated = os.path.join(dst, "model.safetensors")
        assert os.path.isfile(consolidated), "consolidated model.safetensors missing"
        assert not os.path.exists(os.path.join(dst, "model.safetensors.index.json")), \
            "consolidated output must not carry a stale index"
        # every non-weight file from src (config.json, tokenizer files, ...)
        # needs copying too, otherwise dst isn't loadable via from_pretrained
        # on its own
        copied = {f for f in os.listdir(dst) if not f.endswith(".safetensors")}
        assert copied == expected_non_weight, \
            f"non-weight files not copied: {copied} vs {expected_non_weight}"
        loaded = load_file(consolidated)
        assert set(loaded.keys()) == set(tensors.keys()), loaded.keys()
        for k, v in tensors.items():
            assert torch.equal(loaded[k], v), f"tensor {k} changed across consolidation"
        print("  OK (2 shards consolidated into one model.safetensors, "
              "bit-exact tensors, non-weight files copied)")

        # --max-shard-bytes: the sharded-output path, not covered by the
        # pytest suite. the tiny checkpoint is ~6.5MB with the 50257x32 fp32
        # embedding alone at ~6.4MB, so a 3MB cap forces >= 2 shards. a
        # single tensor bigger than the cap still lands in its own seperate
        # shard (write_sharded never splits a tensor)
        dst2 = os.path.join(td, "dst_sharded")
        sys.argv = ["export_safetensors.py", "--src", src, "--dst", dst2,
                    "--max-shard-bytes", "3000000"]
        try:
            main()
        finally:
            sys.argv = old_argv

        with open(os.path.join(dst2, "model.safetensors.index.json")) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        assert len(shard_files) >= 2, f"expected >= 2 shards, got {shard_files}"
        assert set(index["weight_map"].keys()) == set(tensors.keys())
        assert not os.path.exists(os.path.join(dst2, "model.safetensors")), \
            "--max-shard-bytes must not also write a consolidated file"
        resharded = {}
        for fname in shard_files:
            resharded.update(load_file(os.path.join(dst2, fname)))
        assert set(resharded.keys()) == set(tensors.keys())
        for k, v in tensors.items():
            assert torch.equal(resharded[k], v), f"tensor {k} changed across re-sharding"
        print(f"  OK (--max-shard-bytes 3000000 -> {len(shard_files)} shards + "
              f"index.json, bit-exact tensors)")

    print("\n[selftest] All checks passed (no GPU required).")


if __name__ == "__main__":
    main()
