#!/usr/bin/env python3
"""Generate a two-stage variable-shape PaiNN timing configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--input-xyz", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--potential-type", required=True, choices=("NNP", "NNP_csr"))
    parser.add_argument("--max-edges", required=True, type=int)
    parser.add_argument("--seed", type=int, default=2026071101)
    parser.add_argument("--temperature-k", type=float, default=1500.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--warmup-fs", type=float, default=1000.0)
    parser.add_argument("--measure-fs", type=float, default=9000.0)
    parser.add_argument("--thermo-interval-steps", type=int, default=100)
    args = parser.parse_args()
    if args.thermo_interval_steps <= 0:
        raise ValueError("--thermo-interval-steps must be positive")

    def simulation(duration_fs: float, initialize_velocities: bool) -> dict:
        return {
            "dt": args.dt_fs,
            "simulation_time": duration_fs,
            "ensemble": {
                "type": "NVT",
                "temperature": args.temperature_k,
                "thermostat": "Nose-Hoover",
                "tau": 1.0,
                "initialize_velocities": initialize_velocities,
                "temperature_dof": "3N-3",
                "thermostat_dof": "3N-3",
                "remove_com_drift_interval": 128,
            },
            "use_graph": False,
        }

    config = {
        "meta": {
            "name": args.name,
            "unit": "metal",
            "seed": args.seed,
            "benchmark_role": "painn_variable_csr_10ps",
            "warmup_fs": args.warmup_fs,
            "measure_fs": args.measure_fs,
        },
        "common_settings": {
            "atoms": {"mode": "from_file", "format": "xyz", "path": str(args.input_xyz)},
            "cell": {"type": "cubic"},
            "interactions": {
                "cell_list": False,
                "neighbour_list": {"cutoff": 5.0, "margin": 1.0, "max_neighbours": 1000},
                "potentials": {
                    "type": args.potential_type,
                    "cutoff": 5.0,
                    "max_edges": args.max_edges,
                    "model_path": str(args.model),
                },
            },
        },
        "steps": [
            {
                "name": "warmup_1ps",
                "simulation": simulation(args.warmup_fs, True),
                "observer": {"type": "linear", "interval": args.thermo_interval_steps},
            },
            {
                "name": "measurement_9ps",
                "step": "reset",
                "simulation": simulation(args.measure_fs, False),
                "observer": {"type": "linear", "interval": args.thermo_interval_steps},
            },
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
