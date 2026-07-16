#!/usr/bin/env python3
"""Hydra feasibility check for compiling the exact MD-only PaiNN wrapper."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from export_simplegnn_painn_for_md import (
    PainnMDInferenceWrapper,
    _extract_state_dict,
    _load_energy_baseline_lookup,
    _torch_load,
)
from validate_painn_paired_parity import read_paired_graph


def elapsed_ms(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], steps: int) -> float:
    for _ in range(3):
        model(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        model(*inputs)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--energy-baseline-json", required=True, type=Path)
    parser.add_argument("--simplegnn-root", required=True, type=Path)
    parser.add_argument("--xyz", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-atoms", type=int, default=96)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--natom-basis", type=int, default=128)
    parser.add_argument("--n-radial", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    parser.add_argument("--num-interactions", type=int, default=3)
    parser.add_argument("--radial-type", default="gaussian")
    parser.add_argument("--envelope-type", default="smoothstep")
    args = parser.parse_args()

    sys.path.insert(0, str(args.simplegnn_root.resolve()))
    from simplegnn.painn import Painn

    device = torch.device("cuda")
    base = Painn(
        natom_basis=args.natom_basis,
        n_radial=args.n_radial,
        cutoff=args.cutoff,
        epsilon=args.epsilon,
        num_interactions=args.num_interactions,
        radial_type=args.radial_type,
        envelope_type=args.envelope_type,
    )
    checkpoint = _torch_load(args.checkpoint, device)
    base.load_state_dict(_extract_state_dict(checkpoint))
    base.to(device).eval()
    baseline = _load_energy_baseline_lookup(args.energy_baseline_json).to(device)
    wrapper = PainnMDInferenceWrapper(base, baseline).to(device).eval()

    z, edge_index, edge_weight = read_paired_graph(args.xyz, args.max_atoms, args.cutoff)
    inputs = (z.to(device), edge_index.to(device), edge_weight.to(device))
    eager_energy, eager_forces = wrapper(*inputs)

    result: dict[str, object] = {
        "torch_version": torch.__version__,
        "atoms": int(z.shape[0]),
        "edges": int(edge_index.shape[1]),
        "status": "fail",
    }
    try:
        started = time.monotonic()
        compiled = torch.compile(wrapper, fullgraph=True, dynamic=True)
        compiled_energy, compiled_forces = compiled(*inputs)
        compile_seconds = time.monotonic() - started
        energy_difference = float(torch.abs(eager_energy - compiled_energy))
        force_difference = float(torch.max(torch.abs(eager_forces - compiled_forces)))
        eager_ms = elapsed_ms(wrapper, inputs, args.steps)
        compiled_ms = elapsed_ms(compiled, inputs, args.steps)
        result.update(
            {
                "status": "pass",
                "compile_seconds": compile_seconds,
                "energy_difference_ev": energy_difference,
                "max_force_difference_ev_a": force_difference,
                "eager_ms_per_call": eager_ms,
                "compiled_ms_per_call": compiled_ms,
                "speedup": eager_ms / compiled_ms,
                "strict_parity_pass": energy_difference <= 1e-4
                and force_difference <= 1e-5,
            }
        )
    except Exception as error:  # noqa: BLE001 - the diagnostic records compiler failures.
        result["error_type"] = type(error).__name__
        result["error"] = str(error)

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
