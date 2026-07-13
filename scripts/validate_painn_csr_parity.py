#!/usr/bin/env python3
"""Compare scatter and CSR MD-only PaiNN artifacts on one identical graph."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
ATOMIC_NUMBERS = {"Li": 3, "O": 8, "Na": 11, "Si": 14, "K": 19}


def read_csr_graph(
    path: Path,
    max_atoms: int,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = min(int(lines[0]), max_atoms)
    lattice_match = LATTICE_RE.search(lines[1])
    if lattice_match is None:
        raise ValueError(f"missing Lattice in {path}")
    lattice = [float(value) for value in lattice_match.group(1).split()]
    if len(lattice) != 9 or any(lattice[index] != 0.0 for index in (1, 2, 3, 5, 6, 7)):
        raise ValueError("CSR parity helper requires an orthorhombic cell")
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
    mask = (distance < cutoff) & (distance > 0.0)
    edge_index = mask.nonzero(as_tuple=False).transpose(0, 1).contiguous()
    edge_weight = displacement[mask].transpose(0, 1).contiguous()
    if edge_index.shape[1] == 0:
        raise ValueError("representative graph has no edges")

    counts = torch.bincount(edge_index[0], minlength=natoms)
    offsets = torch.zeros(natoms + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return z, edge_index, edge_weight, offsets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scatter", required=True, type=Path)
    parser.add_argument("--csr", required=True, type=Path)
    parser.add_argument("--xyz", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-atoms", type=int, default=96)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--energy-tolerance-ev", type=float, default=1e-4)
    parser.add_argument("--force-tolerance-ev-a", type=float, default=1e-5)
    args = parser.parse_args()

    torch.set_num_threads(1)
    z, edge_index, edge_weight, offsets = read_csr_graph(
        args.xyz,
        args.max_atoms,
        args.cutoff,
    )
    scatter = torch.jit.load(str(args.scatter), map_location="cpu").eval()
    csr = torch.jit.load(str(args.csr), map_location="cpu").eval()
    scatter_energy, scatter_forces = scatter(z, edge_index, edge_weight)
    csr_energy, csr_forces = csr(z, edge_index, edge_weight, offsets)
    energy_difference = float(torch.abs(scatter_energy - csr_energy))
    force_difference = float(torch.max(torch.abs(scatter_forces - csr_forces)))
    result = {
        "status": "pass"
        if energy_difference <= args.energy_tolerance_ev
        and force_difference <= args.force_tolerance_ev_a
        else "fail",
        "atoms": int(z.shape[0]),
        "edges": int(edge_index.shape[1]),
        "segments": int(offsets.shape[0] - 1),
        "energy_difference_ev": energy_difference,
        "max_force_difference_ev_a": force_difference,
        "energy_tolerance_ev": args.energy_tolerance_ev,
        "force_tolerance_ev_a": args.force_tolerance_ev_a,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
