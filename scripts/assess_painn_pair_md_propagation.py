#!/usr/bin/env python3
"""Assess short-MD pair trajectories against a repeated-baseline noise floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FORCE_METRICS = (
    "force_max_abs_ev_per_angstrom",
    "force_rms_ev_per_angstrom",
    "force_p99_9_abs_ev_per_angstrom",
    "force_relative_l2",
)


def assess(args: argparse.Namespace) -> dict:
    static = json.loads(args.static_parity.read_text())
    cases: list[dict] = []
    for ensemble in ("NVE", "NVT"):
        control = json.loads((args.control_dir / f"compare_{ensemble}_baseline_repeat.json").read_text())
        control_maxima = control["maxima"]
        for backend in ("paired_shared", "paired_stacked"):
            candidate = json.loads((args.stepwise_dir / f"compare_{ensemble}_{backend}.json").read_text())
            maxima = candidate["maxima"]
            force_ratios = {
                metric: maxima[metric] / control_maxima[metric]
                if control_maxima[metric] > 0.0
                else (0.0 if maxima[metric] == 0.0 else float("inf"))
                for metric in FORCE_METRICS
            }
            propagation_gate = bool(
                candidate["frame_count_match"]
                and all(frame["structural_match"] and frame["finite"] for frame in candidate["frames"])
                and maxima["position_max_abs_angstrom"] <= args.position_max_angstrom
                and maxima["velocity_max_abs_angstrom_per_fs"] <= args.velocity_max_angstrom_per_fs
            )
            noise_envelope_gate = bool(
                maxima["energy_abs_ev"] <= args.noise_envelope_factor * control_maxima["energy_abs_ev"]
                and all(ratio <= args.noise_envelope_factor for ratio in force_ratios.values())
            )
            cases.append(
                {
                    "ensemble": ensemble,
                    "backend": backend,
                    "accuracy_gate_same_state": static.get("status") == "pass",
                    "propagation_gate": propagation_gate,
                    "noise_envelope_gate": noise_envelope_gate,
                    "noise_envelope_factor": args.noise_envelope_factor,
                    "candidate_maxima": maxima,
                    "baseline_repeat_maxima": control_maxima,
                    "force_metric_ratios_vs_baseline_repeat": force_ratios,
                    "status": "pass"
                    if static.get("status") == "pass" and propagation_gate and noise_envelope_gate
                    else "fail",
                }
            )
    return {
        "method": {
            "same_state_accuracy": "static energy/force parity",
            "trajectory_propagation": "finite/schema plus absolute position/velocity bounds",
            "force_drift": "diagnostic envelope relative to an independently repeated baseline trajectory",
            "note": "Force values on already-diverged coordinates are not reused as a same-state accuracy gate.",
        },
        "thresholds": {
            "position_max_angstrom": args.position_max_angstrom,
            "velocity_max_angstrom_per_fs": args.velocity_max_angstrom_per_fs,
            "noise_envelope_factor": args.noise_envelope_factor,
        },
        "static_parity_status": static.get("status"),
        "cases": cases,
        "status": "pass" if cases and all(case["status"] == "pass" for case in cases) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-parity", required=True, type=Path)
    parser.add_argument("--stepwise-dir", required=True, type=Path)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--position-max-angstrom", type=float, default=1.0e-4)
    parser.add_argument("--velocity-max-angstrom-per-fs", type=float, default=1.0e-5)
    parser.add_argument("--noise-envelope-factor", type=float, default=1.5)
    args = parser.parse_args()
    result = assess(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
