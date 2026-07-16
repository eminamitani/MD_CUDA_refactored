#!/usr/bin/env python3
"""Generate a deterministic short-MD config for PaiNN pair diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_config(args: argparse.Namespace) -> dict:
    ensemble: dict[str, object] = {
        "type": args.ensemble,
        "temperature": args.temperature_k,
        "initialize_velocities": True,
        "rescale_initial_temperature": True,
        "temperature_dof": "3N-3",
        "thermostat_dof": "3N-3",
        "remove_com_drift_interval": 128,
    }
    if args.ensemble == "NVT":
        ensemble.update({"thermostat": "Nose-Hoover", "tau": 1.0})

    return {
        "meta": {
            "name": args.name,
            "unit": "metal",
            "seed": args.seed,
            "diagnostic_role": "painn_pair_stepwise",
        },
        "common_settings": {
            "atoms": {"mode": "from_file", "format": "xyz", "path": str(args.input_xyz)},
            "cell": {"type": "cubic"},
            "interactions": {
                "cell_list": False,
                "neighbour_list": {
                    "cutoff": 5.0,
                    "margin": 1.0,
                    "max_neighbours": 1000,
                },
                "potentials": {
                    "type": "NNP",
                    "cutoff": 5.0,
                    "max_edges": args.max_edges,
                    "model_path": str(args.model),
                },
            },
        },
        "steps": [
            {
                "name": f"{args.ensemble.lower()}_{args.steps}_step_diagnostic",
                "simulation": {
                    "dt": args.dt_fs,
                    "simulation_time": args.steps * args.dt_fs,
                    "ensemble": ensemble,
                    "use_graph": False,
                },
                "observer": {
                    "type": "linear_export_trajectory",
                    "interval": 1,
                    "output_path": str(args.trajectory),
                    "trajectory": {
                        "mode": "transport_base",
                        "fields": ["position", "velocity", "force", "energy"],
                        "coordinates": "wrapped",
                        "format": "extxyz",
                    },
                },
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--input-xyz", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--ensemble", required=True, choices=("NVE", "NVT"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=2026071101)
    parser.add_argument("--max-edges", type=int, default=500000)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.dt_fs <= 0.0:
        raise ValueError("--dt-fs must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trajectory.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_config(args), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
