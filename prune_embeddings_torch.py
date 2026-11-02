#!/usr/bin/env python3
"""PyTorch/ROCm port of an MLX embedding-slicing script — runs on the output
of prune_vocab.py, before expand_model.py. For CUDA/ROCm boxes without MLX
(Apple Silicon only).

Same logic as an MLX original built first for local Apple Silicon experiments,
ported to run on a single AMD MI300X (ROCm) box that can't run MLX. Uses
safetensors.torch instead of mx.load/mx.save_safetensors; PyTorch has native
bfloat16 support so the dtype handling carries over directly.

Usage:
    python3 prune_embeddings_torch.py \\
        --src ./checkpoints/base_12b \\
        --dst ./checkpoints/base_12b_pruned

The embedding tensor's key defaults to the Gemma-4-family layout this was
built and run against, but is a plain CLI flag (--embed-key), not a
hardcoded assumption -- point it at whatever your own checkpoint's
safetensors header actually calls the embedding weight (e.g. plain
`model.embed_tokens.weight` on many non-Gemma architectures) and the tensor
surgery below is otherwise architecture-agnostic: it only ever reads one
named tensor out of the state dict, slices its rows, and writes it back.
Configurable is not the same claim as verified -- this has only actually
been run against the Gemma-4-family key below.
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

# Default matches the Gemma-4-family checkpoints this script has actually been
# run against. Override with --embed-key for a different model family's key
# layout -- see module docstring.
DEFAULT_EMBED_KEY = "model.language_model.embed_tokens.weight"


def load_remap(remap_path: str) -> dict:
    """Load prune_vocab.py's _old_to_new_ids.json. json forces both keys and
    values to strings, so coerce both to int here. the id math below
    (sorting, range checks, row indexing) wants real ints."""
    with open(remap_path) as f:
        return {int(k): int(v) for k, v in json.load(f).items()}


def contiguous_keep_ids(old_to_new: dict) -> tuple:
    """Sorted old ids to keep plus the new vocab size, or SystemExit if the
    remap's new ids are not contiguous 0..N-1 (the row-slicing below just
    assumes they are)."""
    keep_old_ids = sorted(old_to_new.keys())
    new_vocab_size = len(keep_old_ids)
    expected_new_ids = sorted(old_to_new.values())
    if expected_new_ids != list(range(new_vocab_size)):
        raise SystemExit("ERROR: old_to_new id remap is not contiguous 0..N-1 — "
                         "aborting, slicing logic assumes it is.")
    return keep_old_ids, new_vocab_size


def resolve_index(src: str) -> dict:
    """Read model.safetensors.index.json from `src`, or make one up for a
    single-file checkpoint (a lone model.safetensors with no index) via the
    shared helper in expand_model.py."""
    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    from expand_model import synthesize_single_shard_index
    print("[prune_embed] no index.json found -- synthesized one for the single-file checkpoint")
    return synthesize_single_shard_index(src)


def slice_embed_rows(val: torch.Tensor, keep_old_ids: list) -> torch.Tensor:
    """Slice an embedding matrix down to the kept vocab rows. returns a
    contiguous copy since safetensors' save_file refuses non-contiguous
    tensors."""
    keep_idx = torch.tensor(keep_old_ids, dtype=torch.long)
    return val[keep_idx, :].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--embed-key", type=str, default=DEFAULT_EMBED_KEY,
                    help="Safetensors key for the embedding weight tensor to slice. "
                         f"Default ({DEFAULT_EMBED_KEY!r}) matches the Gemma-4-family "
                         "layout this script was built and run against -- for another "
                         "model family, check its safetensors header (e.g. via "
                         "safetensors.safe_open(...).keys()) and pass the real key here.")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU required).")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return

    # Validate required args AFTER the --selftest check.
    if not args.src or not args.dst:
        ap.error("--src and --dst are required (unless --selftest).")

    src, dst = args.src, args.dst
    embed_key = args.embed_key

    remap_path = os.path.join(dst, "_old_to_new_ids.json")
    if not os.path.exists(remap_path):
        raise SystemExit(f"ERROR: {remap_path} not found — run prune_vocab.py against "
                          f"this --dst first, it produces the id remap this script needs.")
    old_to_new = load_remap(remap_path)
    keep_old_ids, new_vocab_size = contiguous_keep_ids(old_to_new)

    # A checkpoint this size can come down from HF as EITHER a sharded
    # multi-file layout (index.json + model-NNNNN-of-MMMMM.safetensors) or a
    # single unsharded model.safetensors with no index at all -- some public
    # Gemma-4-family checkpoints download as the latter (confirmed against a
    # real downloaded checkpoint, not assumed). resolve_index normalizes the
    # single-file case so the rest of the script only ever sees one format.
    index = resolve_index(src)

    embed_shard = index["weight_map"][embed_key]
    all_shards = sorted(set(index["weight_map"].values()))

    print(f"[prune_embed] embed tensor lives in shard: {embed_shard}")
    print(f"[prune_embed] old vocab={len(old_to_new):,} new vocab={new_vocab_size:,}")

    os.makedirs(dst, exist_ok=True)
    new_total_size = 0
    new_param_delta = 0

    for shard in all_shards:
        src_path = os.path.join(src, shard)
        dst_path = os.path.join(dst, shard)

        if shard != embed_shard:
            shutil.copy2(src_path, dst_path)
            new_total_size += os.path.getsize(dst_path)
            print(f"[prune_embed] {shard}: copied unchanged")
            continue

        tensors_in = load_file(src_path)
        tensors_out = {}
        for key, val in tensors_in.items():
            if key == embed_key:
                sliced = slice_embed_rows(val, keep_old_ids)
                tensors_out[key] = sliced
                removed_rows = val.shape[0] - sliced.shape[0]
                new_param_delta -= removed_rows * val.shape[1]
                print(f"[prune_embed] {embed_key}: {tuple(val.shape)} -> {tuple(sliced.shape)}")
            else:
                tensors_out[key] = val

        save_file(tensors_out, dst_path)
        new_total_size += os.path.getsize(dst_path)
        print(f"[prune_embed] {shard}: rewrote with sliced embedding")

    index["metadata"] = index.get("metadata", {})
    index["metadata"]["total_size"] = new_total_size
    old_total_params = index["metadata"].get("total_parameters")
    if old_total_params is not None:
        index["metadata"]["total_parameters"] = old_total_params + new_param_delta
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[prune_embed] wrote index.json  total_size={new_total_size/1024**3:.2f}GB  "
          f"param_delta={new_param_delta:,}")
    print("[prune_embed] done. Next: load-test the result before trusting it for CPT.")


def _self_test():
    """Self-test: a tiny 2-tensor single-file checkpoint (10-row embedding +
    a bias) and a remap JSON in a tempdir, run through main() with
    --embed-key, then check the sliced shape/rows and teh updated index
    (no GPU needed)."""
    import sys
    import tempfile

    print("[selftest] prune_embeddings_torch: slice a tiny checkpoint via main() (no GPU)")
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src")
        dst = os.path.join(td, "dst")
        os.makedirs(src)
        embed_key = "model.embed_tokens.weight"
        bias_key = "model.lm_head.bias"
        # distinct values per row so the row-equality checks mean something
        embed = torch.arange(10 * 4, dtype=torch.float32).reshape(10, 4)
        bias = torch.ones(10)
        save_file({embed_key: embed, bias_key: bias},
                  os.path.join(src, "model.safetensors"))

        # prune_vocab.py's output: keep old ids 0,2,4,6,8 -> new ids 0..4
        os.makedirs(dst)
        keep_old_ids = [0, 2, 4, 6, 8]
        remap = {str(old): new for new, old in enumerate(keep_old_ids)}
        with open(os.path.join(dst, "_old_to_new_ids.json"), "w") as f:
            json.dump(remap, f)

        old_argv = sys.argv
        sys.argv = ["prune_embeddings_torch.py", "--src", src, "--dst", dst,
                    "--embed-key", embed_key]
        try:
            main()
        finally:
            sys.argv = old_argv

        # sliced embedding: 10 -> 5 rows, exactly the kept rows in remap order
        out_tensors = load_file(os.path.join(dst, "model.safetensors"))
        sliced = out_tensors[embed_key]
        assert sliced.shape == (5, 4), tuple(sliced.shape)
        for new_id, old_id in enumerate(keep_old_ids):
            assert torch.equal(sliced[new_id], embed[old_id]), (new_id, old_id)
        # the non-embedding tensor passes through untouched
        assert torch.equal(out_tensors[bias_key], bias)
        print("  OK (embed 10x4 -> 5x4, rows [0,2,4,6,8] kept in order, bias unchanged)")

        # index.json written with total_size updated to the sliced file's size
        # (smaller than the source since 5 of 10 rows got dropped)
        with open(os.path.join(dst, "model.safetensors.index.json")) as f:
            index = json.load(f)
        assert index["weight_map"][embed_key] == "model.safetensors"
        assert index["weight_map"][bias_key] == "model.safetensors"
        dst_size = os.path.getsize(os.path.join(dst, "model.safetensors"))
        assert index["metadata"]["total_size"] == dst_size, index["metadata"]
        assert dst_size < os.path.getsize(os.path.join(src, "model.safetensors"))
        print("  OK (index.json written, total_size updated to the sliced file's size)")

    print("\n[selftest] All checks passed (no GPU required).")


if __name__ == "__main__":
    main()
