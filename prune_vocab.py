#!/usr/bin/env python3
"""Vocab pruning for a Gemma-4-family base model — runs after optional
preprocess_data.py, before prune_embeddings_torch.py.

Removes CJK, Cyrillic, Arabic, Devanagari (Hindi), Mongolian script tokens,
plus tokens carrying distinctive German/Nordic/French/Spanish diacritics
(a, o, u-umlaut, eszett, a/o-ring, ae-ligature, accented vowels, n-tilde,
inverted ?/!, c-cedilla, oe-ligature). This is a CHARACTER-based heuristic,
not real language ID — it catches words that use a distinctive special
character, not plain-Latin words from those languages that happen to overlap
with common English subwords (e.g. German "und" or French "le" survive
untouched).

Decided this way after a naive "target N remaining tokens" plan turned out to
not match the vocab's real composition (ASCII-only tokens alone were far above
a hoped-for target) — going further into the ASCII-only bucket to hit a
specific number would risk cutting code/technical tokens, which is a real
regression class if you're not careful about what "cheap vocab pruning" cuts.

NEVER run this against your pristine base checkpoint in place — keep the
source untouched. This script only ever WRITES to a new output directory;
the source is opened read-only.

Usage:
    python3 prune_vocab.py \\
        --src ./checkpoints/base_12b_bf16 \\
        --dst ./checkpoints/base_12b_pruned

Model-family scope, spelled out plainly: the character-script vocab-dropping
logic above (classify(), REMOVABLE) is genuinely architecture-agnostic — it
only looks at token strings and tokenizer.json, nothing Gemma-specific. But
the config.json post-processing below it is NOT generic, and isn't
pretending to be:

  - The `model_type == "gemma4_unified"` rename to `"gemma4"` is a real,
    narrow fix for one specific transformers-install quirk on Gemma-4-family
    checkpoints (see the code comment at that block for the exact mechanism).
    It only fires when the checkpoint's own model_type matches that string,
    so it's already a no-op for any other model family's config.json --
    running this against a non-Gemma checkpoint simply skips it, it does not
    need a flag to disable.
  - The vocab_size field paths (top-level `vocab_size` plus a nested
    `text_config.vocab_size`) are a Gemma-4-family config layout, but WHICH
    paths get updated is now a `--vocab-size-paths` flag (default: the two
    Gemma-4 paths below), not a hardcoded assumption -- a different model
    family with its own similarly-nested-but-differently-named vocab_size
    field can point this at its own dotted config path(s) instead of needing
    a source edit.

Neither of these two config fixes has been tested against any non-Gemma-4
checkpoint. Configurable is not the same claim as verified.
"""

import argparse
import json
import os
import shutil

CJK_RANGES = [(0x4E00, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7A3)]
CYRILLIC_RANGE = (0x0400, 0x04FF)
ARABIC_RANGE = (0x0600, 0x06FF)
DEVANAGARI_RANGE = (0x0900, 0x097F)
MONGOLIAN_RANGE = (0x1800, 0x18AF)
ROMANCE_GERMANIC_CHARS = set(
    "äöüßåøæÄÖÜÅØÆáéíóúñ¿¡àèìòùâêîôûçœÁÉÍÓÚÑÀÈÌÒÙÂÊÎÔÛÇŒ"
)


def _in_range(c, lo, hi):
    return lo <= ord(c) <= hi


def classify(tok: str) -> str:
    s = tok.replace("▁", "").replace("Ġ", "")  # SentencePiece / GPT2-style space markers
    if not s:
        return "keep"  # empty/space-only tokens are structural, always keep
    if any(any(_in_range(c, lo, hi) for lo, hi in CJK_RANGES) for c in s):
        return "cjk"
    if any(_in_range(c, *CYRILLIC_RANGE) for c in s):
        return "cyrillic"
    if any(_in_range(c, *ARABIC_RANGE) for c in s):
        return "arabic"
    if any(_in_range(c, *DEVANAGARI_RANGE) for c in s):
        return "devanagari_hindi"
    if any(_in_range(c, *MONGOLIAN_RANGE) for c in s):
        return "mongolian_script"
    if any(c in ROMANCE_GERMANIC_CHARS for c in s):
        return "romance_germanic_chars"
    return "keep"


REMOVABLE = {"cjk", "cyrillic", "arabic", "devanagari_hindi", "mongolian_script", "romance_germanic_chars"}


def set_dotted_path(cfg: dict, dotted_path: str, value) -> None:
    """Sets cfg[a][b]...[z] = value given "a.b...z". Every intermediate segment
    must already exist as a dict in cfg (raises KeyError, loudly, rather than
    silently creating a new nested structure the checkpoint's config schema
    never asked for -- a typo'd --vocab-size-paths entry should fail fast, not
    quietly write a field nothing will ever read)."""
    parts = dotted_path.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="Source model dir (read-only, never modified)")
    ap.add_argument("--dst", help="Output dir for the pruned model")
    ap.add_argument("--dry-run", action="store_true", help="Print stats only, write nothing")
    ap.add_argument("--vocab-size-paths", type=str, nargs="+",
                    default=["vocab_size", "text_config.vocab_size"],
                    help="One or more dotted config.json paths to set to the new vocab "
                         "size. Defaults to the two Gemma-4-family locations (top-level "
                         "'vocab_size' plus nested 'text_config.vocab_size') -- see module "
                         "docstring for why both matter on Gemma-4. Pass your own dotted "
                         "path(s) if a different model family nests vocab_size somewhere "
                         "else, e.g. --vocab-size-paths vocab_size some_other.nested.path")
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
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("ERROR: --src and --dst must differ — refusing to overwrite the source in place.")

    tok_path = os.path.join(src, "tokenizer.json")
    with open(tok_path) as f:
        tok = json.load(f)

    vocab: dict[str, int] = tok["model"]["vocab"]
    added_tokens = {t["content"] for t in tok.get("added_tokens", [])}

    # Classify every token. Special/added tokens are ALWAYS kept regardless
    # of script — they're structural (e.g. <pad>, <bos>, vision/audio
    # placeholders), not natural-language content.
    keep_ids: list[int] = []
    drop_ids: list[int] = []
    id_to_tok = {v: k for k, v in vocab.items()}
    for tok_str, tok_id in vocab.items():
        if tok_str in added_tokens:
            keep_ids.append(tok_id)
            continue
        cat = classify(tok_str)
        if cat in REMOVABLE:
            drop_ids.append(tok_id)
        else:
            keep_ids.append(tok_id)

    keep_ids.sort()
    print(f"[prune_vocab] total={len(vocab):,}  keep={len(keep_ids):,}  drop={len(drop_ids):,}")

    if args.dry_run:
        return

    # Build new vocab with remapped contiguous IDs (0..N-1), preserving
    # original relative order so frequency-rank structure isn't disturbed.
    new_vocab: dict[str, int] = {}
    old_to_new: dict[int, int] = {}
    for new_id, old_id in enumerate(keep_ids):
        tok_str = id_to_tok[old_id]
        new_vocab[tok_str] = new_id
        old_to_new[old_id] = new_id

    keep_tok_strs = set(new_vocab.keys())

    # Filter merge rules: a merge (a, b) -> a+b is only valid if a, b, AND
    # the merged result all survive pruning. Order is preserved (BPE merge
    # priority is positional).
    #
    # Merges come in two on-disk formats depending on the tokenizer.json source:
    #   - HF `tokenizers` default: a list of "a b" space-separated STRINGS
    #     (e.g. "Ġ t"). Splitting on the first space gives the two parts, and
    #     the composed token is a+b (no space) — exactly what the vocab stores.
    #   - Older/list format: a 2-element list [a, b].
    # The pre-fix code did `a, b = pair[0], pair[1]`, which on a STRING pair
    # unpacked the first two CHARACTERS — silently gutting the merge table to
    # near-empty (only the rare merge whose first two chars were both surviving
    # single-char vocab entries survived). Handle both formats explicitly.
    old_merges = tok["model"]["merges"]
    new_merges = []
    dropped_merges = 0
    for pair in old_merges:
        if isinstance(pair, str):
            a, b = pair.split(" ", 1)
            # Preserve the original string in the output so the format round-trips
            # exactly for tokenizers that read the string form.
            out_pair = pair
        else:
            a, b = pair[0], pair[1]
            out_pair = pair
        merged = a + b
        if a in keep_tok_strs and b in keep_tok_strs and merged in keep_tok_strs:
            new_merges.append(out_pair)
        else:
            dropped_merges += 1
    print(f"[prune_vocab] merges: {len(old_merges):,} -> {len(new_merges):,}  (dropped {dropped_merges:,})")

    tok["model"]["vocab"] = new_vocab
    tok["model"]["merges"] = new_merges

    # Remap added_tokens ids to the new id space.
    for at in tok.get("added_tokens", []):
        old_id = at["id"]
        if old_id in old_to_new:
            at["id"] = old_to_new[old_id]
        else:
            raise SystemExit(f"ERROR: added/special token {at['content']!r} (id={old_id}) was "
                              f"unexpectedly dropped — this should never happen, special tokens "
                              f"are protected above. Aborting before writing anything.")

    os.makedirs(dst, exist_ok=True)

    with open(os.path.join(dst, "tokenizer.json"), "w") as f:
        json.dump(tok, f, ensure_ascii=False)
    print(f"[prune_vocab] wrote {os.path.join(dst, 'tokenizer.json')}")

    # Copy everything else byte-for-byte EXCEPT tokenizer.json (just written)
    # and the safetensors weight files (handled separately — see
    # prune_embeddings_torch.py, which does the actual tensor-slicing surgery
    # and must run after this script produces old_to_new.json).
    for fname in os.listdir(src):
        if fname in ("tokenizer.json",) or fname.endswith(".safetensors"):
            continue
        s = os.path.join(src, fname)
        d = os.path.join(dst, fname)
        if os.path.isfile(s):
            shutil.copy2(s, d)

    # Update vocab_size in config.json at every path in --vocab-size-paths.
    # On Gemma-4-family checkpoints (the default paths) this MUST be set at
    # BOTH the top level AND inside text_config — discovered via a real
    # load-test failure: mlx_lm's Gemma4UnifiedModelArgs.__post_init__ does
    # `self.text_config["vocab_size"] = self.vocab_size`, where
    # `self.vocab_size` defaults to 262144 if the top-level key is absent.
    # That overwrite CLOBBERS a correctly-set text_config.vocab_size if the
    # top-level key is missing — setting only the nested field (the first
    # version of this script did exactly that) silently reverts to 262144
    # at model-build time, caught by a strict load_weights() shape mismatch.
    # A different model family with a different (or single, unnested)
    # vocab_size layout should pass its own path(s) via --vocab-size-paths
    # rather than relying on this being universal -- it isn't tested against
    # any non-Gemma-4 config.json.
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    for path in args.vocab_size_paths:
        set_dotted_path(cfg, path, len(new_vocab))

    # model_type fix -- confirmed via a real crash: some Gemma-4-family
    # checkpoints ship "gemma4_unified" as the top-level model_type, but a
    # given transformers install's Gemma4Config class may register itself
    # under plain "gemma4" (model_type = "gemma4" in configuration_gemma4.py)
    # -- AutoConfig's CONFIG_MAPPING["gemma4_unified"] lookup then raises
    # KeyError before ever reaching any auto_map/trust_remote_code handling,
    # since that lookup happens first regardless. If the checkpoint's actual
    # structure (text_config/vision_config/audio_config sub-dicts) matches
    # Gemma4Config.sub_configs, renaming this one field is safe. Check your
    # installed transformers version's registered model_type before assuming
    # this applies to your checkpoint.
    if cfg.get("model_type") == "gemma4_unified":
        cfg["model_type"] = "gemma4"
        print("[prune_vocab] config.json model_type 'gemma4_unified' -> 'gemma4' "
              "(matches what this transformers install's Gemma4Config actually registers as)")

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[prune_vocab] config.json vocab_size -> {len(new_vocab):,} "
          f"(paths: {', '.join(args.vocab_size_paths)})")

    # Save the old->new id remap for the embedding-slicing step.
    remap_path = os.path.join(dst, "_old_to_new_ids.json")
    with open(remap_path, "w") as f:
        json.dump({str(k): v for k, v in old_to_new.items()}, f)
    print(f"[prune_vocab] wrote id remap -> {remap_path}")
    print("[prune_vocab] NEXT STEP: run prune_embeddings_torch.py against this dir to slice "
          "the embed_tokens.weight tensor — this dir is NOT yet a loadable model "
          "(safetensors still have the old full-size embedding matrix).")


def _self_test():
    """Self-test: a tiny tokenizer.json (ASCII + CJK tokens, both merge
    formats, added_tokens) + config.json in a tempdir, run through main();
    then check the kept vocab, the filtered merges, the id remap,
    vocab_size at both config paths, and the model_type rename (no GPU
    needed)."""
    import sys
    import tempfile

    import torch
    from safetensors.torch import save_file

    print("[selftest] prune_vocab: tiny tokenizer + config through main()")
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src")
        dst = os.path.join(td, "dst")
        os.makedirs(src)

        # tiny vocab: ASCII tokens to keep, a CJK content token to drop, plus
        # an added/special CJK token (protected, special tokens are alwasy
        # kept regardless of script)
        vocab = {
            "<pad>": 0, "Ġ": 1, "t": 2, "h": 3, "Ġt": 4, "Ġh": 5,
            "世界": 6, "你好": 7, "x": 8,
        }
        # both on-disk merge formats (see main()'s merge-filtering block):
        # HF tokenizers' "a b" strings and the older [a, b] lists
        merges = ["Ġ t", "Ġ h", ["Ġ", "t"], "世 界", "x y", ["世", "界"]]
        tok = {
            "version": "1.0",
            "model": {"type": "BPE", "vocab": vocab, "merges": merges},
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": True},
                {"id": 7, "content": "你好", "special": True},
            ],
        }
        with open(os.path.join(src, "tokenizer.json"), "w") as f:
            json.dump(tok, f, ensure_ascii=False)

        # gemma-4-family config layout: vocab_size at BOTH paths plus the
        # model_type that main() renames
        cfg = {
            "model_type": "gemma4_unified",
            "vocab_size": 9,
            "text_config": {"vocab_size": 9},
        }
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump(cfg, f)
        # weights deliberately NOT copied, that's prune_embeddings_torch.py's
        # job. a real tiny safetensors file proves the contract
        save_file({"dummy.weight": torch.zeros(2, 2)},
                  os.path.join(src, "model.safetensors"))

        old_argv = sys.argv
        sys.argv = ["prune_vocab.py", "--src", src, "--dst", dst]
        try:
            main()
        finally:
            sys.argv = old_argv

        # kept vocab should be the 8 ASCII/structural tokens plus the
        # protected CJK added token, with the CJK content token dropped and
        # ids remapped contiguously in their original relative order
        expected_keep = ["<pad>", "Ġ", "t", "h", "Ġt", "Ġh", "你好", "x"]
        with open(os.path.join(dst, "tokenizer.json")) as f:
            out_tok = json.load(f)
        out_vocab = out_tok["model"]["vocab"]
        assert sorted(out_vocab.keys()) == sorted(expected_keep), \
            sorted(out_vocab.keys())
        assert sorted(out_vocab.values()) == list(range(len(expected_keep)))
        assert out_vocab["<pad>"] == 0 and out_vocab["Ġ"] == 1
        assert out_vocab["你好"] == 6 and out_vocab["x"] == 7
        print("  OK (kept vocab: CJK content token dropped, added CJK token "
              "protected, ids contiguous and order-preserving)")

        # merges: only the ones whose parts AND merged result all survive,
        # each pair keeping its original format (string vs list)
        assert out_tok["model"]["merges"] == ["Ġ t", "Ġ h", ["Ġ", "t"]], \
            out_tok["model"]["merges"]
        print("  OK (merges filtered in both string and list formats)")

        # added_tokens ids get remapped into the new id space
        out_added = {t["content"]: t["id"] for t in out_tok["added_tokens"]}
        assert out_added == {"<pad>": 0, "你好": 6}, out_added
        print("  OK (added_tokens ids remapped)")

        # the id remap for prune_embeddings_torch.py
        with open(os.path.join(dst, "_old_to_new_ids.json")) as f:
            out_remap = json.load(f)
        assert out_remap == {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
                             "5": 5, "7": 6, "8": 7}, out_remap
        print("  OK (_old_to_new_ids.json written with the expected remap)")

        # config.json: vocab_size updated at BOTH default paths, model_type
        # renamed, and no safetensors copied over
        with open(os.path.join(dst, "config.json")) as f:
            out_cfg = json.load(f)
        assert out_cfg["vocab_size"] == 8, out_cfg["vocab_size"]
        assert out_cfg["text_config"]["vocab_size"] == 8
        assert out_cfg["model_type"] == "gemma4", out_cfg["model_type"]
        assert not os.path.exists(os.path.join(dst, "model.safetensors"))
        print("  OK (vocab_size updated at both config paths, model_type "
              "'gemma4_unified' -> 'gemma4', safetensors not copied)")

    print("\n[selftest] All checks passed (no GPU required).")


if __name__ == "__main__":
    main()
