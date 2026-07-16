#!/usr/bin/env python3
"""Compile PaiNN energy while retaining the strict create_graph=True force path."""

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


class ExactEnergyModule(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, e0_lookup: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("e0_lookup", e0_lookup)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight_e3: torch.Tensor,
    ) -> torch.Tensor:
        node_scalar = self.base.embedding(z)
        node_vector = torch.zeros(
            (node_scalar.shape[0], 3, node_scalar.shape[1]),
            dtype=node_scalar.dtype,
            device=node_scalar.device,
        )
        for message, mixing in zip(self.base.message_layers, self.base.mixing_layers):
            node_scalar, node_vector = message(
                node_scalar,
                node_vector,
                edge_index,
                edge_weight_e3,
            )
            node_scalar, node_vector = mixing(node_scalar, node_vector)
        atom_energy = self.base.output(node_scalar)
        baseline = self.e0_lookup[z].to(dtype=atom_energy.dtype).sum()
        return atom_energy.sum() + baseline


def energy_force(
    energy_model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    z, edge_index, edge_weight = inputs
    edge_weight_e3 = edge_weight.transpose(0, 1).contiguous()
    edge_weight_e3 = edge_weight_e3.detach().requires_grad_(True)
    energy = energy_model(z, edge_index, edge_weight_e3)
    edge_gradient = torch.autograd.grad(
        [energy],
        [edge_weight_e3],
        create_graph=True,
    )[0].detach()
    index_i = edge_index[0]
    index_j = edge_index[1]
    force_i = torch.zeros((z.shape[0], 3), device=z.device, dtype=edge_gradient.dtype)
    force_j = torch.zeros_like(force_i)
    force_i.scatter_add_(0, index_i.unsqueeze(1).expand_as(edge_gradient), edge_gradient)
    force_j.scatter_add_(0, index_j.unsqueeze(1).expand_as(edge_gradient), -edge_gradient)
    return energy, (force_i + force_j).transpose(0, 1).contiguous()


def elapsed_ms(
    energy_model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    steps: int,
) -> float:
    for _ in range(3):
        energy_force(energy_model, inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        energy_force(energy_model, inputs)
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
    args = parser.parse_args()

    sys.path.insert(0, str(args.simplegnn_root.resolve()))
    from simplegnn.painn import Painn

    device = torch.device("cuda")
    base = Painn(
        natom_basis=128,
        n_radial=32,
        cutoff=5.0,
        epsilon=1e-7,
        num_interactions=3,
        radial_type="gaussian",
        envelope_type="smoothstep",
    )
    checkpoint = _torch_load(args.checkpoint, device)
    base.load_state_dict(_extract_state_dict(checkpoint))
    base.to(device).eval()
    baseline = _load_energy_baseline_lookup(args.energy_baseline_json).to(device)
    reference = PainnMDInferenceWrapper(base, baseline).to(device).eval()
    exact_energy = ExactEnergyModule(base, baseline).to(device).eval()

    z, edge_index, edge_weight = read_paired_graph(args.xyz, args.max_atoms, 5.0)
    inputs = (z.to(device), edge_index.to(device), edge_weight.to(device))
    reference_energy, reference_forces = reference(*inputs)
    eager_energy, eager_forces = energy_force(exact_energy, inputs)
    result: dict[str, object] = {
        "torch_version": torch.__version__,
        "atoms": int(z.shape[0]),
        "edges": int(edge_index.shape[1]),
        "eager_energy_difference_ev": float(torch.abs(reference_energy - eager_energy)),
        "eager_max_force_difference_ev_a": float(
            torch.max(torch.abs(reference_forces - eager_forces))
        ),
        "status": "fail",
    }
    try:
        started = time.monotonic()
        compiled_energy = torch.compile(exact_energy, fullgraph=True, dynamic=True)
        candidate_energy, candidate_forces = energy_force(compiled_energy, inputs)
        result["compile_seconds"] = time.monotonic() - started
        result["compiled_energy_difference_ev"] = float(
            torch.abs(reference_energy - candidate_energy)
        )
        result["compiled_max_force_difference_ev_a"] = float(
            torch.max(torch.abs(reference_forces - candidate_forces))
        )
        result["strict_parity_pass"] = bool(
            result["compiled_energy_difference_ev"] <= 1e-4
            and result["compiled_max_force_difference_ev_a"] <= 1e-5
        )
        result["eager_ms_per_call"] = elapsed_ms(exact_energy, inputs, args.steps)
        result["compiled_ms_per_call"] = elapsed_ms(compiled_energy, inputs, args.steps)
        result["speedup"] = result["eager_ms_per_call"] / result["compiled_ms_per_call"]
        result["status"] = "pass" if result["strict_parity_pass"] else "parity_fail"
    except Exception as error:  # noqa: BLE001 - compiler diagnostics are the output.
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
