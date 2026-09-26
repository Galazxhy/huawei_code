"""Reproduce the V005 resource allocation cases from the supplied handoffs."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
for stage in ("Q3_1", "Q3_2", "Q3_3"):
    sys.path.insert(0, str(ROOT / stage))

from q3_model_contract import Q3ModelContract, sha256  # noqa: E402
from q3_optimizer import OptimizerConfig  # noqa: E402
from run_q3_experiments import BUDGETS, FORMS, MODES, run_job  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", action="store_true", help="run all 11 budgets, 5 contexts, 3 costs, and 4 modes")
    parser.add_argument("--budget", type=float, default=1e22, help="FLOPs for a single case")
    parser.add_argument("--context", type=int, default=4096, help="C7 context length for a single case")
    parser.add_argument("--form", choices=FORMS, default="power", help="quality cost function")
    parser.add_argument("--mode", choices=MODES, action="append", help="repeat to compare decision modes")
    parser.add_argument("--output", type=Path, default=Path("results/reproduction.jsonl"))
    args = parser.parse_args()

    model = Q3ModelContract()
    manifest = model.source_manifest()
    manifest[str(Path("q3_v005") / "run.py")] = sha256(ROOT / "run.py")
    if args.grid:
        if args.mode:
            parser.error("--mode applies only to a single case")
        cases = itertools.product(BUDGETS, model.contexts, FORMS, MODES)
    else:
        if args.budget <= 0 or args.context not in model.contexts:
            parser.error("budget must be positive and context must be one of the C7 values")
        cases = itertools.product((args.budget,), (args.context,), (args.form,),
                                  args.mode or ("joint_reduced_hull", "fixed_p_q_baseline"))

    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must be inside this package")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"output already exists: {output}")

    failures = 0
    with output.open("x", encoding="utf-8") as stream:
        for budget, context, form, mode in cases:
            job = {"bridge": "ratio", "kappa": 1.0,
                   "config": vars(OptimizerConfig()), "scenario": "main",
                   "budget": budget, "context": context, "form": form, "mode": mode}
            record = run_job(job, manifest)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            result = record["result"]
            summary = {"status": record["status"], "budget": budget, "context": context,
                       "form": form, "mode": mode}
            if result is not None:
                summary.update(mean_loss=result["mean_loss"], N=result["N_params"],
                               D=result["D_tokens"], Q=result["Q"],
                               budget_margin=result["budget_margin_flops"])
            else:
                summary["error"] = record["error"]
                failures += 1
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Saved results to {output.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
