from __future__ import annotations

import sys

from ..catalog import recommend_model
from ..config import load_config, save_config
from ..doctor import is_ready, run_checks
from ..hardware import detect_hardware
from ._download_model import _download_model
from ._print_checks import _print_checks


def cmd_setup(args) -> int:
    hw = detect_hardware()

    # 1. Readiness.
    print("Checking readiness ...")
    checks = run_checks(hw)
    _print_checks(checks)
    if not is_ready(checks) and not args.force:
        print(
            "\nNot ready — resolve the ✗ items above, or re-run with --force to "
            "configure anyway (download/smoke-test will likely fail).",
            file=sys.stderr,
        )
        return 1

    # 2. Choose a model.
    model = args.model
    if model:
        print(f"\nUsing requested model: {model}")
    else:
        available = hw.recommended_working_set_bytes or hw.total_ram_bytes
        if not available:
            print("error: could not determine available memory to recommend a model.", file=sys.stderr)
            return 2
        rec = recommend_model(available, safety_fraction=args.safety)
        if rec is None:
            print(
                "error: no catalog model fits this machine's memory budget. "
                "Specify a tiny model explicitly with --model.",
                file=sys.stderr,
            )
            return 1
        model = rec.repo_id
        fit = rec.estimate()
        print(f"\nRecommended: {rec.name}")
        print(f"  {model}")
        print(f"  ~{rec.params / 1e9:.1f}B params @ {rec.quant_bits}-bit, weights {fit.human()}")

    # 3. Download (unless skipped).
    if args.no_download:
        print("\nSkipping download (--no-download).")
    else:
        print(f"\nDownloading {model} (this can take a while the first time) ...")
        try:
            path = _download_model(model)
            print(f"  cached at {path}")
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            print(f"error: download failed: {e}", file=sys.stderr)
            return 1

    # 4. Persist as the default model.
    cfg = load_config()
    cfg.model = model
    cfg_path = save_config(cfg)
    print(f"\nSet default model -> {model}\n  ({cfg_path})")

    # 5. Smoke test.
    if args.no_smoke_test:
        print("\nSkipping smoke test (--no-smoke-test).")
    elif args.no_download:
        print("\nSkipping smoke test (no weights downloaded).")
    else:
        print("\nRunning a smoke-test generation ...")
        try:
            from ..runner import GenerationConfig, MLXNotAvailableError, ModelRunner

            runner = ModelRunner.load(model, trust_remote_code=args.trust_remote_code)
            out = runner.generate(
                prompt="Reply with a single short sentence to confirm you are working.",
                config=GenerationConfig(max_tokens=32, temperature=0.0),
            )
            print(f"  model said: {out.strip()[:200]}")
        except MLXNotAvailableError as e:
            print(f"  smoke test skipped: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            print(f"  smoke test failed: {e}", file=sys.stderr)
            return 1

    print("\n✓ Setup complete. Try:  mlx-runner chat")
    return 0
