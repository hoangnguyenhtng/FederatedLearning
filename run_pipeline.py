"""
Small command wrapper for the project workflow.

Examples:
    python run_pipeline.py --mode train
    python run_pipeline.py --mode evaluate
    python run_pipeline.py --mode api
    python run_pipeline.py --mode dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(args: list[str]) -> int:
    """Run a subprocess from the project root."""
    print("\n" + "=" * 70)
    print("Running:", " ".join(args))
    print("=" * 70 + "\n")
    completed = subprocess.run(args, cwd=PROJECT_ROOT)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Federated recommendation project runner")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "api", "dashboard", "all"],
        default="train",
        help="Workflow step to run",
    )
    args = parser.parse_args()

    steps: list[list[str]] = []

    if args.mode in {"train", "all"}:
        steps.append([sys.executable, "src/training/federated_training_pipeline.py"])

    if args.mode in {"evaluate", "all"}:
        steps.append([sys.executable, "src/training/evaluate_federated_model.py"])

    if args.mode == "api":
        steps.append([sys.executable, "src/api/fastapi_app.py"])

    if args.mode == "dashboard":
        steps.append(["streamlit", "run", "src/dashboard/explainable_ui.py"])

    # `all` intentionally stops after evaluation. API/dashboard are long-running
    # processes and should be started in separate terminals for demo.
    for step in steps:
        code = run_command(step)
        if code != 0:
            return code

    print("\n✅ Completed:", args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
