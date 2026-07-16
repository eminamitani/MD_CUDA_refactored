#!/usr/bin/env python3
"""Compare MD_CUDA extxyz trajectories with float32-aware diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ENERGY_RE = re.compile(r"(?:^|\s)energy=([0-9.eE+-]+)")
TIME_RE = re.compile(r"(?:^|\s)time_fs=([0-9.eE+-]+)")


def parse_extxyz(path: Path) -> list[dict]:
    frames: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            n_atoms = int(line)
            comment = handle.readline().strip()
            energy_match = ENERGY_RE.search(comment)
            time_match = TIME_RE.search(comment)
            species: list[str] = []
            position: list[float] = []
            velocity: list[float] = []
            force: list[float] = []
            for _ in range(n_atoms):
                fields = handle.readline().split()
                if len(fields) != 10:
                    raise ValueError(f"Expected species+9 values in {path}, got {len(fields)}")
                species.append(fields[0])
                values = [float(value) for value in fields[1:]]
                position.extend(values[0:3])
                velocity.extend(values[3:6])
                force.extend(values[6:9])
            frames.append(
                {
                    "n_atoms": n_atoms,
                    "species": species,
                    "energy": float(energy_match.group(1)) if energy_match else None,
                    "time_fs": float(time_match.group(1)) if time_match else None,
                    "position": position,
                    "velocity": velocity,
                    "force": force,
                }
            )
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def diagnostics(reference: list[float], candidate: list[float]) -> dict[str, float]:
    if len(reference) != len(candidate):
        raise ValueError("Array lengths differ")
    differences = [abs(left - right) for left, right in zip(reference, candidate)]
    squares = [(left - right) ** 2 for left, right in zip(reference, candidate)]
    reference_squares = [value * value for value in reference]
    ordered = sorted(differences)
    percentile_index = max(0, math.ceil(0.999 * len(ordered)) - 1)
    norm = math.sqrt(sum(reference_squares))
    return {
        "max_abs": max(differences, default=0.0),
        "rms": math.sqrt(sum(squares) / len(squares)) if squares else 0.0,
        "p99_9_abs": ordered[percentile_index] if ordered else 0.0,
        "relative_l2": math.sqrt(sum(squares)) / norm if norm else math.sqrt(sum(squares)),
    }


def compare(args: argparse.Namespace) -> dict:
    reference = parse_extxyz(args.reference)
    candidate = parse_extxyz(args.candidate)
    frame_count_match = len(reference) == len(candidate)
    count = min(len(reference), len(candidate))
    frame_results: list[dict] = []
    first_failure: dict | None = None
    for index in range(count):
        ref_frame, cand_frame = reference[index], candidate[index]
        structural_match = bool(
            ref_frame["n_atoms"] == cand_frame["n_atoms"]
            and ref_frame["species"] == cand_frame["species"]
            and ref_frame["time_fs"] == cand_frame["time_fs"]
        )
        energy_difference = (
            abs(ref_frame["energy"] - cand_frame["energy"])
            if ref_frame["energy"] is not None and cand_frame["energy"] is not None
            else math.inf
        )
        position = diagnostics(ref_frame["position"], cand_frame["position"])
        velocity = diagnostics(ref_frame["velocity"], cand_frame["velocity"])
        force = diagnostics(ref_frame["force"], cand_frame["force"])
        finite = all(
            math.isfinite(value)
            for value in (
                energy_difference,
                *position.values(),
                *velocity.values(),
                *force.values(),
            )
        )
        passed = bool(
            structural_match
            and finite
            and energy_difference <= args.energy_max_ev
            and position["max_abs"] <= args.position_max_angstrom
            and velocity["max_abs"] <= args.velocity_max_angstrom_per_fs
            and force["max_abs"] <= args.force_max_ev_per_angstrom
            and force["rms"] <= args.force_rms_ev_per_angstrom
        )
        result = {
            "frame": index,
            "time_fs": ref_frame["time_fs"],
            "structural_match": structural_match,
            "finite": finite,
            "energy_abs_ev": energy_difference,
            "position": position,
            "velocity": velocity,
            "force": force,
            "pass": passed,
        }
        frame_results.append(result)
        if not passed and first_failure is None:
            first_failure = result

    def maximum(field: str, metric: str) -> float:
        return max((frame[field][metric] for frame in frame_results), default=math.inf)

    summary = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_frames": len(reference),
        "candidate_frames": len(candidate),
        "frame_count_match": frame_count_match,
        "thresholds": {
            "energy_max_ev": args.energy_max_ev,
            "position_max_angstrom": args.position_max_angstrom,
            "velocity_max_angstrom_per_fs": args.velocity_max_angstrom_per_fs,
            "force_max_ev_per_angstrom": args.force_max_ev_per_angstrom,
            "force_rms_ev_per_angstrom": args.force_rms_ev_per_angstrom,
        },
        "maxima": {
            "energy_abs_ev": max((frame["energy_abs_ev"] for frame in frame_results), default=math.inf),
            "position_max_abs_angstrom": maximum("position", "max_abs"),
            "velocity_max_abs_angstrom_per_fs": maximum("velocity", "max_abs"),
            "force_max_abs_ev_per_angstrom": maximum("force", "max_abs"),
            "force_rms_ev_per_angstrom": maximum("force", "rms"),
            "force_p99_9_abs_ev_per_angstrom": maximum("force", "p99_9_abs"),
            "force_relative_l2": maximum("force", "relative_l2"),
        },
        "first_failure": first_failure,
        "status": "pass" if frame_count_match and first_failure is None else "fail",
        "frames": frame_results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--energy-max-ev", type=float, default=1.0e-4)
    parser.add_argument("--position-max-angstrom", type=float, default=1.0e-4)
    parser.add_argument("--velocity-max-angstrom-per-fs", type=float, default=1.0e-5)
    parser.add_argument("--force-max-ev-per-angstrom", type=float, default=5.0e-5)
    parser.add_argument("--force-rms-ev-per-angstrom", type=float, default=1.0e-5)
    args = parser.parse_args()
    summary = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
