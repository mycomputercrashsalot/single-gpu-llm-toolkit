#!/usr/bin/env python3
"""Export a causal-LM checkpoint to ONNX.

Exports only the base model (not MTP modules) for inference. Requires
`transformers` and a checkpoint that can be loaded with
`AutoModelForCausalLM.from_pretrained`.
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="Source checkpoint directory.")
    ap.add_argument("--dst", help="Output directory or .onnx file path.")
    # NOTE: dest names (batch/max_seq_len) match the dummy_input construction
    # below, which was never updated when an earlier pass renamed these to
    # batch_size/seq_length -- that left main with args.batch/args.max_seq_len
    # referencing undefined attributes. Fixed by keeping dest names in sync
    # with actual usage, found while merging this branch in.
    ap.add_argument("--max-seq-len", "--seq-length", type=int, default=128,
                    dest="max_seq_len", help="Dummy input sequence length (alias: --seq-length).")
    ap.add_argument("--batch", "--batch-size", type=int, default=1, dest="batch",
                    help="Dummy input batch size (alias: --batch-size).")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "fp8"], default="fp32")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU required).")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return

    # Validate required args AFTER the --selftest check.
    if not args.src or not args.dst:
        ap.error("--src and --dst are required (unless --selftest).")

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("transformers is required for ONNX export") from exc

    import torch
    from runtime import DTYPE_MAP

    torch_dtype = DTYPE_MAP[args.dtype]

    print(f"[export_onnx] loading {args.src} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.src,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.randint(
        0, model.config.vocab_size, (args.batch, args.max_seq_len), device=device
    )

    dst = args.dst
    if os.path.isdir(dst) or not dst.endswith(".onnx"):
        os.makedirs(dst, exist_ok=True)
        dst = os.path.join(dst, "model.onnx")

    # Wrap the model so torch.onnx.export gets a plain Tensor→Tensor function.
    # HF CausalLM models return a namedtuple (CausalLMOutputWithCrossAttentions),
    # which torch 2.13's Dynamo-based ONNX exporter (the new default since
    # torch 2.9) can't trace through. The wrapper extracts .logits, making the
    # graph a plain Tensor→Tensor that exports cleanly. The legacy TorchScript
    # tracer (dynamo=False) also fails on GPT-2's dynamic attention control
    # flow, so this wrapper is needed regardless of exporter mode.
    class LogitsOnlyWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
        def forward(self, input_ids):
            return self.model(input_ids=input_ids).logits

    export_model = LogitsOnlyWrapper(model)
    export_model.eval()

    print(f"[export_onnx] exporting to {dst} ...")
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            dst,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=18,
        )
    print("[export_onnx] done.")


def _self_test():
    """Self-test: run main() on a tiny real GPT-2 checkpoint and validate the
    exported graph with onnx.checker (CPU, no GPU needed).

    onnx/onnxscript get guarded the same way the pytest suite's ONNX test
    guards them (tests/test_all.py importorskips both): if either one is
    missing, print the reason and exit 0, don't fail."""
    import sys
    import tempfile

    for dep in ("onnx", "onnxscript"):
        try:
            __import__(dep)
        except ImportError as exc:
            print(f"[selftest] SKIP: {dep} is not installed ({exc}) -- the ONNX "
                  f"export path needs onnx + onnxscript. Install both to "
                  f"exercise the full path. Exiting 0.")
            return

    import onnx
    from transformers import GPT2Config, GPT2LMHeadModel

    print("[selftest] export_onnx: main() on a tiny GPT-2 checkpoint (CPU, no GPU)")
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "tiny_gpt2")
        # same tiny config as tests/test_all.py's _build_tiny_gpt2_checkpoint
        cfg = GPT2Config(vocab_size=50257, n_positions=128, n_embd=32,
                         n_layer=2, n_head=4)
        GPT2LMHeadModel(cfg).save_pretrained(src)

        dst = os.path.join(td, "model.onnx")
        old_argv = sys.argv
        sys.argv = ["export_onnx.py", "--src", src, "--dst", dst,
                    "--dtype", "fp32", "--seq-length", "8"]
        try:
            main()
        finally:
            sys.argv = old_argv

        assert os.path.isfile(dst), f"{dst} was not produced"
        onnx.checker.check_model(onnx.load(dst))
        print(f"  OK ({dst} exists and passes onnx.checker.check_model)")

    print("\n[selftest] All checks passed (CPU-only, no GPU required).")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", default=False)
    args, _ = ap.parse_known_args()
    if args.selftest:
        _self_test()
    else:
        main()


if __name__ == "__main__":
    main_cli()
