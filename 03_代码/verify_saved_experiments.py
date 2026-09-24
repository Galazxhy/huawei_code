"""Re-evaluate saved registered predictions without rerunning optimizers."""
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "03_结果分析/analyze_v2_evidence.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = ROOT / "results/复现清单.json"
    manifest = json.loads(manifest_path.read_text())
    checked = {}
    for entry in manifest["input_files"]:
        path = Path(entry["path"])
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("manifest input outside this version")
        if sha(path) != entry["sha256"]:
            raise ValueError(f"changed release input: {path}")
        checked[str(path)] = entry["sha256"]
    spec = importlib.util.spec_from_file_location("release_analysis", ANALYSIS)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    inputs = analysis.Inputs(ROOT)
    evaluator = analysis.Evaluator(ROOT, inputs)
    raw_paths = set()
    for relative in manifest["key_parameters"]["strict_reports"]:
        report = inputs.read(ROOT / relative)
        if report["publication_blocked"] or report["source_manifest"] != evaluator.manifest:
            raise ValueError("strict evidence is incomplete or stale")
        for name, checksum in report["input_sha256"].items():
            path = Path(name)
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError("source provenance leaves version")
            if str(path) in checked and checked[str(path)] != checksum:
                raise ValueError("conflicting source hashes")
            if str(path) not in checked and sha(path) != checksum:
                raise ValueError(f"changed experiment source: {path}")
            checked[str(path)] = checksum
            if "/results/formal/" in str(path) and path.name != "batch_status.json":
                raw_paths.add(path)
        directory = ROOT / Path(relative).parent
        for name, checksum in report["output_sha256"].items():
            path = directory / name
            if sha(path) != checksum:
                raise ValueError(f"changed analysis output: {path}")
            checked[str(path)] = checksum
    count, max_gap, unique = 0, 0., set()
    for path in sorted(raw_paths):
        record = inputs.read(path)
        if record["status"] != "PASS" or record["source_manifest"] != evaluator.manifest:
            raise ValueError("saved registered result is not current/admissible")
        evaluator.validate(record["job"], record["result"])
        value = evaluator.evaluate(record["job"], record["result"])
        # The evaluator also recomputes domain values, costs, support and bounds.
        if isinstance(value, tuple):
            value = value[0]
        max_gap = max(max_gap, abs(value["mean_loss"]-record["result"]["mean_loss"]))
        unique.add(evaluator.runner.job_key(record["job"]))
        count += 1
    if len(unique) != 15960:
        raise ValueError(f"incomplete mathematical configuration coverage: {len(unique)}")
    if inputs.changed() or evaluator.base.source_manifest() != evaluator.manifest:
        raise ValueError("inputs changed during re-evaluation")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "qa" / ("saved_experiment_reproduction_"+timestamp+".json")
    result = {"status": "PASS", "scope": "saved predictions, constraints and evidence arithmetic; no new optimization",
              "registered_unique_configurations": len(unique), "physical_records_re_evaluated": count,
              "hashed_files": len(checked), "maximum_mean_loss_difference": max_gap,
              "reproduction_manifest_sha256": sha(manifest_path), "script_sha256": sha(Path(__file__)),
              "python": sys.executable, "source_manifest": evaluator.manifest}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"status": result["status"], "records": count, "unique": len(unique), "report": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
