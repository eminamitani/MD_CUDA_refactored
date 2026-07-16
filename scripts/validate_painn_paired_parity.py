#!/usr/bin/env python3
"""Compare directed and pair-folded MD-only PaiNN artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
ATOMIC_NUMBERS = {"Li": 3, "O": 8, "Na": 11, "Si": 14, "K": 19}


def read_paired_graph(
    path: Path,
    max_atoms: int,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = min(int(lines[0]), max_atoms)
    lattice_match = LATTICE_RE.search(lines[1])
    if lattice_match is None:
        raise ValueError(f"missing Lattice in {path}")
    lattice = [float(value) for value in lattice_match.group(1).split()]
    if len(lattice) != 9 or any(lattice[index] != 0.0 for index in (1, 2, 3, 5, 6, 7)):
        raise ValueError("paired parity helper requires an orthorhombic cell")
    box = torch.tensor([lattice[0], lattice[4], lattice[8]], dtype=torch.float32)

    species: list[int] = []
    positions: list[list[float]] = []
    for line in lines[2 : 2 + natoms]:
        fields = line.split()
        species.append(ATOMIC_NUMBERS[fields[0]])
        positions.append([float(value) for value in fields[1:4]])
    z = torch.tensor(species, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float32)

    displacement = pos[None, :, :] - pos[:, None, :]
    displacement -= box * torch.round(displacement / box)
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    upper = torch.triu(torch.ones((natoms, natoms), dtype=torch.bool), diagonal=1)
    pair_mask = upper & (distance < cutoff)
    pair_index = pair_mask.nonzero(as_tuple=False).transpose(0, 1).contiguous()
    pair_weight = displacement[pair_mask].transpose(0, 1).contiguous()
    if pair_index.shape[1] == 0:
        raise ValueError("representative graph has no pairs")
    reverse_index = torch.stack((pair_index[1], pair_index[0]), dim=0)
    edge_index = torch.cat((pair_index, reverse_index), dim=1)
    edge_weight = torch.cat((pair_weight, -pair_weight), dim=1)
    return z, edge_index, edge_weight


def force_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = candidate - reference
    flattened = torch.abs(difference).reshape(-1)
    relative_l2 = torch.linalg.vector_norm(difference) / torch.clamp_min(
        torch.linalg.vector_norm(reference),
        1e-12,
    )
    return {
        "max_force_difference_ev_a": float(torch.max(torch.abs(difference))),
        "rms_force_difference_ev_a": float(torch.sqrt(torch.mean(difference.square()))),
        "p99_9_force_difference_ev_a": float(torch.quantile(flattened, 0.999)),
        "relative_l2_force_difference": float(relative_l2),
        "reference_max_force_ev_a": float(torch.max(torch.abs(reference))),
        "reference_rms_force_ev_a": float(torch.sqrt(torch.mean(reference.square()))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directed", required=True, type=Path)
    parser.add_argument("--paired", required=True, type=Path)
    parser.add_argument("--stacked", type=Path)
    parser.add_argument("--xyz", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-atoms", type=int, default=96)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--displacement-scale", type=float, default=0.02)
    parser.add_argument("--energy-tolerance-ev", type=float, default=1e-4)
    parser.add_argument("--force-tolerance-ev-a", type=float, default=5e-5)
    parser.add_argument("--force-rms-tolerance-ev-a", type=float, default=1e-5)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)
    z, edge_index, edge_weight = read_paired_graph(args.xyz, args.max_atoms, args.cutoff)
    z = z.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    directed = torch.jit.load(str(args.directed), map_location=device).eval()
    candidates = {"paired_shared": torch.jit.load(str(args.paired), map_location=device).eval()}
    if args.stacked is not None:
        candidates["paired_stacked"] = torch.jit.load(
            str(args.stacked), map_location=device
        ).eval()

    if args.variants < 1:
        raise ValueError("--variants must be positive")
    pair_count = edge_index.shape[1] // 2
    pair_weight = edge_weight[:, :pair_count]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    edge_weight_variants = [edge_weight]
    strain_scales = (0.99, 1.01, 1.02)
    for variant_index in range(1, args.variants):
        if variant_index <= len(strain_scales):
            varied_pair = pair_weight * strain_scales[variant_index - 1]
        else:
            noise = torch.randn(
                pair_weight.shape,
                generator=generator,
                dtype=pair_weight.dtype,
            ) * args.displacement_scale
            varied_pair = pair_weight.cpu() + noise
            varied_pair = varied_pair.to(device)
        edge_weight_variants.append(torch.cat((varied_pair, -varied_pair), dim=1))

    comparisons: dict[str, dict[str, object]] = {}
    passed = True
    for name, model in candidates.items():
        energy_differences: list[float] = []
        max_force_differences: list[float] = []
        rms_force_differences: list[float] = []
        p99_9_force_differences: list[float] = []
        relative_l2_force_differences: list[float] = []
        reference_max_forces: list[float] = []
        reference_rms_forces: list[float] = []
        finite = True
        failed_variants: list[int] = []
        for variant_index, variant_weight in enumerate(edge_weight_variants):
            reference_energy, reference_forces = directed(z, edge_index, variant_weight)
            energy, forces = model(z, edge_index, variant_weight)
            metrics = force_metrics(reference_forces, forces)
            energy_difference = float(torch.abs(reference_energy - energy))
            energy_differences.append(energy_difference)
            max_force_differences.append(metrics["max_force_difference_ev_a"])
            rms_force_differences.append(metrics["rms_force_difference_ev_a"])
            p99_9_force_differences.append(metrics["p99_9_force_difference_ev_a"])
            relative_l2_force_differences.append(metrics["relative_l2_force_difference"])
            reference_max_forces.append(metrics["reference_max_force_ev_a"])
            reference_rms_forces.append(metrics["reference_rms_force_ev_a"])
            variant_finite = bool(torch.isfinite(energy) and torch.isfinite(forces).all())
            finite = finite and variant_finite
            if (
                energy_difference > args.energy_tolerance_ev
                or metrics["max_force_difference_ev_a"] > args.force_tolerance_ev_a
                or metrics["rms_force_difference_ev_a"] > args.force_rms_tolerance_ev_a
                or not variant_finite
            ):
                failed_variants.append(variant_index)
        candidate_passed = not failed_variants
        comparisons[name] = {
            "max_energy_difference_ev": max(energy_differences),
            "max_force_difference_ev_a": max(max_force_differences),
            "max_rms_force_difference_ev_a": max(rms_force_differences),
            "max_p99_9_force_difference_ev_a": max(p99_9_force_differences),
            "max_relative_l2_force_difference": max(relative_l2_force_differences),
            "max_reference_force_ev_a": max(reference_max_forces),
            "max_reference_rms_force_ev_a": max(reference_rms_forces),
            "finite": finite,
            "failed_variant_indices": failed_variants,
            "passed": candidate_passed,
        }
        passed = passed and candidate_passed

    result = {
        "status": "pass" if passed else "fail",
        "atoms": int(z.shape[0]),
        "pairs": int(edge_index.shape[1] // 2),
        "directed_edges": int(edge_index.shape[1]),
        "variants": len(edge_weight_variants),
        "energy_tolerance_ev": args.energy_tolerance_ev,
        "force_tolerance_ev_a": args.force_tolerance_ev_a,
        "force_rms_tolerance_ev_a": args.force_rms_tolerance_ev_a,
        "comparisons": comparisons,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
