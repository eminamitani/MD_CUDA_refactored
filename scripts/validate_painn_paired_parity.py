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
    return {
        "max_force_difference_ev_a": float(torch.max(torch.abs(difference))),
        "rms_force_difference_ev_a": float(torch.sqrt(torch.mean(difference.square()))),
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
    parser.add_argument("--energy-tolerance-ev", type=float, default=1e-4)
    parser.add_argument("--force-tolerance-ev-a", type=float, default=1e-5)
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

    reference_energy, reference_forces = directed(z, edge_index, edge_weight)
    comparisons: dict[str, dict[str, float | bool]] = {}
    passed = True
    for name, model in candidates.items():
        energy, forces = model(z, edge_index, edge_weight)
        metrics = force_metrics(reference_forces, forces)
        energy_difference = float(torch.abs(reference_energy - energy))
        candidate_passed = (
            energy_difference <= args.energy_tolerance_ev
            and metrics["max_force_difference_ev_a"] <= args.force_tolerance_ev_a
            and bool(torch.isfinite(energy))
            and bool(torch.isfinite(forces).all())
        )
        comparisons[name] = {
            "energy_difference_ev": energy_difference,
            **metrics,
            "finite": bool(torch.isfinite(energy) and torch.isfinite(forces).all()),
            "passed": candidate_passed,
        }
        passed = passed and candidate_passed

    result = {
        "status": "pass" if passed else "fail",
        "atoms": int(z.shape[0]),
        "pairs": int(edge_index.shape[1] // 2),
        "directed_edges": int(edge_index.shape[1]),
        "energy_tolerance_ev": args.energy_tolerance_ev,
        "force_tolerance_ev_a": args.force_tolerance_ev_a,
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
