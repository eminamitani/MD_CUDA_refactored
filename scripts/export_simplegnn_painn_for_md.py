#!/usr/bin/env python3
"""Export a simplegnn PaiNN checkpoint for the MD_CUDA_refactored NNP backend.

The MD code calls ordinary TorchScript models as:

    model(x, edge_index, edge_weight)

where `edge_weight` is stored as `[3, E]` and the returned forces must be laid
out as `[3, N]` so the C++ side can copy x/y/z force blocks directly.

The simplegnn PaiNN implementation uses:

    model(Z, edge_index, edge_weight, batch)

where `edge_weight` is `[E, 3]` and forces are returned as `[N, 3]`.
This exporter wraps the trained model and saves a TorchScript module with the
MD-compatible interface. CSR exports additionally accept `offsets` with shape
`[N + 1]` and use source-sorted segment reductions for PaiNN message updates.
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
    """Legacy adapter from simplegnn PaiNN to the MD NNP TorchScript ABI."""

    def __init__(self, base: torch.nn.Module, e0_lookup: Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("e0_lookup", e0_lookup)

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
        e0_lookup = self.e0_lookup.to(device=energy.device)
        baseline = e0_lookup[Z.long()].to(dtype=energy.dtype).sum()

        # MD expects a contiguous force buffer [fx_0..fx_N, fy_0.., fz_0..].
        forces_soa = forces.reshape(-1, 3).transpose(0, 1).contiguous()
        return energy.sum() + baseline, forces_soa


class PainnMDInferenceWrapper(torch.nn.Module):
    """PaiNN force-only inference path for the MD NNP TorchScript ABI.

    Training PaiNN evaluates batch aggregation and a virial-like tensor and
    keeps the force graph for force-loss backpropagation.  MD needs only total
    energy and first-derivative forces, so this wrapper reuses the trained
    modules, omits batch/virial work, and detaches the derivative before it is
    returned to MD.
    """

    def __init__(self, base: torch.nn.Module, e0_lookup: Tensor):
        super().__init__()
        self.embedding = base.embedding
        self.message_layers = base.message_layers
        self.mixing_layers = base.mixing_layers
        self.output = base.output
        self.register_buffer("e0_lookup", e0_lookup)

        # Only the displacement tensor needs gradients during MD inference.
        for parameter in self.parameters():
            parameter.requires_grad_(False)

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

        edge_weight_e3 = edge_weight.transpose(0, 1).contiguous()
        edge_weight_e3 = edge_weight_e3.detach().requires_grad_(True)

        node_scalar = self.embedding(Z)
        node_vector = torch.zeros(
            (node_scalar.shape[0], 3, node_scalar.shape[1]),
            dtype=node_scalar.dtype,
            device=node_scalar.device,
        )
        for message, mixing in zip(self.message_layers, self.mixing_layers):
            node_scalar, node_vector = message(
                node_scalar,
                node_vector,
                edge_index,
                edge_weight_e3,
            )
            node_scalar, node_vector = mixing(node_scalar, node_vector)

        atom_energy = self.output(node_scalar)
        diff_energy = torch.autograd.grad(
            [atom_energy.sum()],
            [edge_weight_e3],
            # Some PaiNN backward kernels take a numerically different path
            # when create_graph=False.  Gen6 force parity exceeds the strict
            # 1e-5 eV/A export tolerance in that mode, so retain the legacy
            # derivative path and detach immediately after the first
            # derivative instead of returning a force graph to MD.
            create_graph=True,
        )[0]
        assert diff_energy is not None
        diff_energy = diff_energy.detach()

        force_i = torch.zeros(
            (node_scalar.shape[0], 3),
            dtype=atom_energy.dtype,
            device=atom_energy.device,
        )
        force_j = torch.zeros_like(force_i)
        index_i = edge_index[0].unsqueeze(1)
        index_j = edge_index[1].unsqueeze(1)
        force_i = torch.scatter_add(force_i, 0, index_i.expand_as(diff_energy), diff_energy)
        force_j = torch.scatter_add(force_j, 0, index_j.expand_as(diff_energy), -diff_energy)
        forces_soa = (force_i + force_j).transpose(0, 1).contiguous()

        baseline = self.e0_lookup[Z].to(dtype=atom_energy.dtype).sum()
        return atom_energy.sum() + baseline, forces_soa


class PainnMDCSRMessageLayer(torch.nn.Module):
    """PaiNN message layer using CSR source segments instead of scatter_add."""

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.natom_basis = int(base.natom_basis)
        self.radial = base.radial
        self.envelope = base.envelope
        self.interaction_context_network = base.interaction_context_network
        self.filter_network = base.filter_network

    def forward(
        self,
        node_scalar: Tensor,
        node_vector: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        offsets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        context = self.interaction_context_network(node_scalar)
        distances = torch.norm(edge_weight, dim=-1)
        directions = edge_weight / distances.unsqueeze(-1)
        basis = self.radial(distances)
        cutoff = self.envelope(distances)
        filters = self.filter_network(basis) * cutoff

        index_j = edge_index[1]
        context_j = context[index_j]
        vector_j = node_vector[index_j]
        messages = filters * context_j
        delta_scalar, delta_radial, delta_vector = torch.split(
            messages,
            self.natom_basis,
            dim=-1,
        )

        scalar_update = torch.segment_reduce(
            delta_scalar,
            "sum",
            offsets=offsets,
        )
        delta_radial = delta_radial.unsqueeze(1)
        delta_vector = delta_vector.unsqueeze(1)
        vector_messages = (
            delta_radial * directions[..., None]
            + delta_vector * vector_j
        )
        vector_update = torch.segment_reduce(
            vector_messages,
            "sum",
            offsets=offsets,
        )
        return node_scalar + scalar_update, node_vector + vector_update


class PainnMDCSRInferenceWrapper(torch.nn.Module):
    """MD-only PaiNN forward using a four-input CSR TorchScript ABI."""

    def __init__(self, base: torch.nn.Module, e0_lookup: Tensor):
        super().__init__()
        self.embedding = base.embedding
        self.message_layers = torch.nn.ModuleList(
            [PainnMDCSRMessageLayer(message) for message in base.message_layers]
        )
        self.mixing_layers = base.mixing_layers
        self.output = base.output
        self.register_buffer("e0_lookup", e0_lookup)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        Z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        offsets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if Z.dtype != torch.long:
            Z = Z.long()
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        if offsets.dtype != torch.long:
            offsets = offsets.long()

        edge_weight_e3 = edge_weight.transpose(0, 1).contiguous()
        edge_weight_e3 = edge_weight_e3.detach().requires_grad_(True)

        node_scalar = self.embedding(Z)
        node_vector = torch.zeros(
            (node_scalar.shape[0], 3, node_scalar.shape[1]),
            dtype=node_scalar.dtype,
            device=node_scalar.device,
        )
        for message, mixing in zip(self.message_layers, self.mixing_layers):
            node_scalar, node_vector = message(
                node_scalar,
                node_vector,
                edge_index,
                edge_weight_e3,
                offsets,
            )
            node_scalar, node_vector = mixing(node_scalar, node_vector)

        atom_energy = self.output(node_scalar)
        diff_energy = torch.autograd.grad(
            [atom_energy.sum()],
            [edge_weight_e3],
            create_graph=True,
        )[0]
        assert diff_energy is not None
        diff_energy = diff_energy.detach()

        force_i = torch.segment_reduce(
            diff_energy,
            "sum",
            offsets=offsets,
        )
        force_j = torch.zeros_like(force_i)
        index_j = edge_index[1].unsqueeze(1)
        force_j = torch.scatter_add(
            force_j,
            0,
            index_j.expand_as(diff_energy),
            -diff_energy,
        )
        forces_soa = (force_i + force_j).transpose(0, 1).contiguous()

        baseline = self.e0_lookup[Z].to(dtype=atom_energy.dtype).sum()
        return atom_energy.sum() + baseline, forces_soa


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


def _load_energy_baseline_lookup(path: Path | None) -> Tensor:
    lookup_size = 119
    if path is None:
        return torch.zeros(lookup_size, dtype=torch.float32)
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("kind") in (None, "none"):
        return torch.zeros(lookup_size, dtype=torch.float32)
    if metadata.get("kind") != "element_ls":
        raise ValueError(f"Unsupported energy baseline kind: {metadata.get('kind')!r}")
    e0_by_z = {int(z): float(value) for z, value in metadata.get("e0_eV", {}).items()}
    if not e0_by_z:
        raise ValueError(f"{path} has no e0_eV entries")
    lookup_size = max(lookup_size, max(e0_by_z) + 1)
    lookup = torch.zeros(lookup_size, dtype=torch.float32)
    for atomic_number, value in e0_by_z.items():
        lookup[atomic_number] = value
    return lookup


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
    parser.add_argument("--energy-baseline-json", type=Path, default=None)
    parser.add_argument(
        "--forward-mode",
        choices=("md", "legacy"),
        default="md",
        help="Use the MD-only inference path (default) or the legacy training forward path.",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=("scatter", "csr"),
        default="scatter",
        help="Use ordinary scatter aggregation (default) or the four-input CSR MD ABI.",
    )
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

    e0_lookup = _load_energy_baseline_lookup(args.energy_baseline_json).to(device)
    if args.aggregation_mode == "csr" and args.forward_mode != "md":
        raise ValueError("--aggregation-mode csr requires --forward-mode md")
    if args.aggregation_mode == "csr":
        wrapper = PainnMDCSRInferenceWrapper(base, e0_lookup).to(device).eval()
    elif args.forward_mode == "md":
        wrapper = PainnMDInferenceWrapper(base, e0_lookup).to(device).eval()
    else:
        wrapper = PainnMDNNPWrapper(base, e0_lookup).to(device).eval()
    scripted = torch.jit.script(wrapper)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
