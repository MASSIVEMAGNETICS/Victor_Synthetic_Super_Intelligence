#!/usr/bin/env python3
"""
Guided setup CLI/TUI fallback for Victor.

Provides a step-by-step interactive flow to:
- Validate Python environment
- Install dependencies (delegates to bootstrap.py)
- Run a smoke check
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).parent
BOOTSTRAP = ROOT / "bootstrap.py"


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _prompt_bool(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def run_step(title: str, fn: Callable[[], bool]) -> bool:
    print(f"\n=== {title} ===")
    try:
        return fn()
    except KeyboardInterrupt:
        print("Cancelled by user.")
        return False
    except Exception as exc:
        print(f"Step failed: {exc}")
        return False


def step_validate_python() -> bool:
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    return sys.version_info >= (3, 8)


def step_install() -> bool:
    code, out, err = _run([sys.executable, str(BOOTSTRAP), "--allow-missing"])
    print(out)
    if err:
        print(err)
    return code == 0


def step_smoke() -> bool:
    code, out, err = _run([sys.executable, str(BOOTSTRAP), "--smoke-only", "--allow-missing"])
    print(out)
    if err:
        print(err)
    return code == 0


def main() -> int:
    print("Victor Guided Setup")
    print("-------------------")
    if not _prompt_bool("Continue with setup?", True):
        return 1

    steps = [
        ("Validate Python environment", step_validate_python),
        ("Install dependencies", step_install),
        ("Run smoke check", step_smoke),
    ]

    results = []
    for title, fn in steps:
        ok = run_step(title, fn)
        results.append((title, ok))
        if not ok and not _prompt_bool("Continue despite failure?", False):
            break

    print("\nSummary:")
    print(json.dumps({title: ok for title, ok in results}, indent=2))
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
