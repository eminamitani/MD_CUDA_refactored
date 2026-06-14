#!/usr/bin/env python3
"""Export a simplegnn PaiNN checkpoint for the MD_CUDA_refactored NNP backend.

The MD code calls TorchScript models as:

    model(x, edge_index, edge_weight)

where `edge_weight` is stored as `[3, E]` and the returned forces must be laid
out as `[3, N]` so the C++ side can copy x/y/z force blocks directly.

The simplegnn PaiNN implementation uses:

    model(Z, edge_index, edge_weight, batch)

where `edge_weight` is `[E, 3]` and forces are returned as `[N, 3]`.
This exporter wraps the trained model and saves a TorchScript module with the
MD-compatible interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Tuple

import torch
from torch import Tensor


class PainnMDNNPWrapper(torch.nn.Module):
    """Adapter from simplegnn PaiNN to the MD NNP TorchScript ABI."""

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    def forward(
        self,
        Z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if Z.dtype != torch.long:
            Z = Z.long()
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        # MD stores displacement vectors as x/y/z blocks: [3, E].
        # PaiNN expects one 3-vector per edge: [E, 3].
        edge_weight_e3 = edge_weight.transpose(0, 1).contiguous()
        edge_weight_e3 = edge_weight_e3.detach().requires_grad_(True)

        batch = torch.zeros(Z.shape[0], dtype=torch.long, device=Z.device)
        energy, forces, _ = self.base(Z, edge_index, edge_weight_e3, batch)

        # MD expects a contiguous force buffer [fx_0..fx_N, fy_0.., fz_0..].
        forces_soa = forces.reshape(-1, 3).transpose(0, 1).contiguous()
        return energy.sum(), forces_soa


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise TypeError(
        "Checkpoint must be a state_dict, a dict containing model_state_dict/state_dict, "
        "or a torch.nn.Module."
    )


def _parse_radial_kwargs(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("--radial-kwargs-json must decode to a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a simplegnn PaiNN checkpoint as an MD-compatible TorchScript NNP model.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Input PaiNN .pth checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Output TorchScript .pt path")
    parser.add_argument(
        "--simplegnn-root",
        type=Path,
        default=None,
        help="Path to the simplegnn_version2 repository if simplegnn is not installed",
    )
    parser.add_argument("--device", default="cpu", help="Device used while exporting, default: cpu")
    parser.add_argument("--natom-basis", type=int, default=60)
    parser.add_argument("--n-radial", type=int, default=40)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--radial-type", default="gauss")
    parser.add_argument("--envelope-type", default="smoothstep")
    parser.add_argument(
        "--radial-kwargs-json",
        default="{}",
        help='JSON object passed as radial_kwargs, for example \'{"start": 0.0}\'',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.simplegnn_root is not None:
        sys.path.insert(0, str(args.simplegnn_root.resolve()))

    from simplegnn.painn import Painn

    device = torch.device(args.device)
    radial_kwargs = _parse_radial_kwargs(args.radial_kwargs_json)
    base = Painn(
        natom_basis=args.natom_basis,
        n_radial=args.n_radial,
        cutoff=args.cutoff,
        epsilon=args.epsilon,
        num_interactions=args.num_interactions,
        radial_type=args.radial_type,
        envelope_type=args.envelope_type,
        radial_kwargs=radial_kwargs,
    )

    checkpoint = _torch_load(args.checkpoint, device)
    base.load_state_dict(_extract_state_dict(checkpoint))
    base.to(device).eval()

    wrapper = PainnMDNNPWrapper(base).to(device).eval()
    scripted = torch.jit.script(wrapper)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
