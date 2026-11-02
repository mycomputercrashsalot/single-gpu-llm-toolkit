#!/usr/bin/env python3
"""Export a checkpoint to GGUF format for llama.cpp.

This tool is a thin wrapper around the official llama.cpp conversion script.
If `llama.cpp/convert_hf_to_gguf.py` is not available, it prints the exact
command you need to run manually.
"""

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="Source checkpoint directory (HF format).")
    ap.add_argument("--dst", help="Output GGUF file path.")
    ap.add_argument("--outtype", default="f16",
                    help="GGUF quantization type (e.g. f16, q4_k_m, q8_0).")
    ap.add_argument("--convert-script",
                    help="Path to llama.cpp convert_hf_to_gguf.py. Auto-detected if not set.")
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="Run built-in self-test (no GPU/llama.cpp required).")
    args = ap.parse_args()

    if args.selftest:
        _self_test()
        return

    # Validate required args AFTER the --selftest check.
    if not args.src or not args.dst:
        ap.error("--src and --dst are required (unless --selftest).")

    convert_script = args.convert_script
    if convert_script is None:
        candidates = [
            "llama.cpp/convert_hf_to_gguf.py",
            "../llama.cpp/convert_hf_to_gguf.py",
            "../../llama.cpp/convert_hf_to_gguf.py",
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                convert_script = cand
                break

    if convert_script is None or not os.path.isfile(convert_script):
        print(
            "[export_gguf] ERROR: could not find llama.cpp/convert_hf_to_gguf.py.\n"
            "Please clone llama-cpp (https://github.com/ggml-org/llama.cpp) and run:\n"
            f"  python3 llama.cpp/convert_hf_to_gguf.py {args.src} "
            f"--outfile {args.dst} --outtype {args.outtype}",
            file=sys.stderr,
        )
        sys.exit(1)

    # NOTE: convert_hf_to_gguf.py takes the source model directory as a
    # positional argument ("model", nargs="?"), NOT a --src flag. Verified
    # against the live upstream script (ggml-org/llama.cpp master,
    # convert_hf_to_gguf.py) during code review -- the previous `--src`
    # invocation would have failed with an argparse error (unrecognized
    # arguments / missing required "model" positional) on any real
    # llama.cpp checkout.
    cmd = [
        sys.executable, convert_script,
        args.src,
        "--outfile", args.dst,
        "--outtype", args.outtype,
    ]
    print(f"[export_gguf] running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print("[export_gguf] done.")


def _self_test():
    """Self-test: point main() at a --convert-script that doesn't exist and
    assert it exits 1 after printing the manual command for exactly those
    args (the argv-forwarding contract). no convert script gets faked here,
    the not-found error path is the real behavior under test."""
    import contextlib
    import io
    import os
    import tempfile

    print("[selftest] export_gguf: missing convert-script -> exit 1 + manual command")
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src")
        os.makedirs(src)
        dst = os.path.join(td, "model.gguf")
        missing_script = os.path.join(td, "no_such_convert_hf_to_gguf.py")
        assert not os.path.exists(missing_script)

        old_argv = sys.argv
        sys.argv = ["export_gguf.py", "--src", src, "--dst", dst,
                    "--outtype", "q8_0", "--convert-script", missing_script]
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                main()
            raise AssertionError(
                "main() should have exited 1 for a nonexistent --convert-script")
        except SystemExit as e:
            assert e.code == 1, f"expected exit code 1, got {e.code}"
        finally:
            sys.argv = old_argv

        # the manual command has to come out with OUR argv forwarded into it
        # (positional src, --outfile dst, --outtype) so a user can copy-paste
        # it against a real llama.cpp checkout
        msg = err.getvalue()
        assert "convert_hf_to_gguf.py" in msg, msg
        assert src in msg, f"manual command must forward --src, got:\n{msg}"
        assert f"--outfile {dst}" in msg, f"manual command must forward --dst, got:\n{msg}"
        assert "--outtype q8_0" in msg, f"manual command must forward --outtype, got:\n{msg}"
        print("  OK (exit code 1; manual command printed with src/dst/outtype forwarded)")

    print("\n[selftest] All checks passed (no GPU/llama.cpp required).")


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
