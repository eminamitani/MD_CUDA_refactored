#!/usr/bin/env python3
"""Aggregate the four-backend variable-shape CSR A100 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


BACKENDS = ("legacy", "md_baseline", "md_sync", "md_csr")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--baseline-binary", required=True, type=Path)
    parser.add_argument("--candidate-binary", required=True, type=Path)
    parser.add_argument("--legacy-model", required=True, type=Path)
    parser.add_argument("--md-scatter-model", required=True, type=Path)
    parser.add_argument("--md-csr-model", required=True, type=Path)
    parser.add_argument("--input-xyz", required=True, type=Path)
    parser.add_argument("--parity-json", required=True, type=Path)
    parser.add_argument("--nve-validation-json", required=True, type=Path)
    parser.add_argument("--sync-profile-json", required=True, type=Path)
    parser.add_argument("--required-speedup", type=float, default=1.25)
    parser.add_argument("--thermo-limit-fraction", type=float, default=0.01)
    args = parser.parse_args()

    runs = [json.loads(path.read_text()) for path in sorted(args.run_root.glob("*/run_summary.json"))]
    grouped = {
        backend: sorted(
            [run for run in runs if run["backend"] == backend],
            key=lambda run: run["repeat"],
        )
        for backend in BACKENDS
    }
    counts_ok = all(len(grouped[backend]) == 3 for backend in BACKENDS)
    all_runs_valid = counts_ok and all(run.get("valid") for run in runs)
    medians = {
        backend: {
            "steps_per_second": statistics.median(run["steps_per_second"] for run in values) if values else None,
            "ns_per_day": statistics.median(run["ns_per_day"] for run in values) if values else None,
            "measurement_seconds": statistics.median(run["measurement_seconds"] for run in values) if values else None,
        }
        for backend, values in grouped.items()
    }
    legacy_sps = medians["legacy"]["steps_per_second"]
    speedup_vs_legacy = {
        backend: medians[backend]["steps_per_second"] / legacy_sps
        for backend in BACKENDS[1:]
    } if legacy_sps else {backend: None for backend in BACKENDS[1:]}
    md_baseline_sps = medians["md_baseline"]["steps_per_second"]
    sync_speedup_vs_md_baseline = (
        medians["md_sync"]["steps_per_second"] / md_baseline_sps
        if md_baseline_sps else None
    )
    csr_speedup_vs_md_sync = (
        medians["md_csr"]["steps_per_second"] / medians["md_sync"]["steps_per_second"]
        if medians["md_sync"]["steps_per_second"] else None
    )

    thermo_differences: list[float] = []
    for repeat in (1, 2, 3):
        reference = next((run for run in grouped["legacy"] if run["repeat"] == repeat), None)
        if reference is None:
            continue
        for backend in BACKENDS[1:]:
            candidate = next((run for run in grouped[backend] if run["repeat"] == repeat), None)
            if candidate is None:
                continue
            for field in ("mean_total_energy_ev", "mean_temperature_k"):
                denominator = max(abs(reference[field]), 1e-12)
                thermo_differences.append(abs(candidate[field] - reference[field]) / denominator)
    max_thermo_difference = max(thermo_differences, default=None)
    thermo_pass = bool(
        max_thermo_difference is not None
        and max_thermo_difference <= args.thermo_limit_fraction
    )

    parity = json.loads(args.parity_json.read_text())
    nve_validation = json.loads(args.nve_validation_json.read_text())
    sync_profile = json.loads(args.sync_profile_json.read_text())
    parity_pass = parity.get("status") == "pass"
    nve_pass = nve_validation.get("status") == "pass"
    sync_profile_pass = sync_profile.get("status") == "pass"
    final_speedup = speedup_vs_legacy["md_csr"]
    accepted = bool(
        all_runs_valid
        and parity_pass
        and nve_pass
        and sync_profile_pass
        and thermo_pass
        and final_speedup is not None
        and final_speedup >= args.required_speedup
    )
    sync_optimization_accepted = bool(
        all_runs_valid
        and parity_pass
        and sync_profile_pass
        and sync_speedup_vs_md_baseline is not None
        and sync_speedup_vs_md_baseline >= 1.01
    )

    summary = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "accepted": accepted,
        "sync_optimization_accepted": sync_optimization_accepted,
        "required_speedup": args.required_speedup,
        "source_commit": args.source_commit,
        "run_counts": {backend: len(values) for backend, values in grouped.items()},
        "all_runs_valid": all_runs_valid,
        "parity": parity,
        "parity_pass": parity_pass,
        "nve_validation": nve_validation,
        "nve_validation_pass": nve_pass,
        "sync_profile": sync_profile,
        "sync_profile_pass": sync_profile_pass,
        "thermodynamics_limit_fraction": args.thermo_limit_fraction,
        "max_thermodynamics_relative_difference": max_thermo_difference,
        "thermodynamics_non_degradation_pass": thermo_pass,
        "median": medians,
        "speedup_vs_legacy": speedup_vs_legacy,
        "sync_speedup_vs_md_baseline": sync_speedup_vs_md_baseline,
        "csr_speedup_vs_md_sync": csr_speedup_vs_md_sync,
        "artifacts_sha256": {
            "baseline_binary": sha256(args.baseline_binary),
            "candidate_binary": sha256(args.candidate_binary),
            "legacy_model": sha256(args.legacy_model),
            "md_scatter_model": sha256(args.md_scatter_model),
            "md_csr_model": sha256(args.md_csr_model),
            "input_xyz": sha256(args.input_xyz),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
