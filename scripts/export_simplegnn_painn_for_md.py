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
import torch.nn.functional as F
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


class PainnMDFunctionalSmoothstepEnvelope(torch.nn.Module):
    """Storage-free equivalent of simplegnn's scripted smoothstep envelope."""

    def __init__(self, rc: float, order: int):
        super().__init__()
        self.rc = float(rc)
        self.order = int(order)

    def forward(self, r: Tensor) -> Tensor:
        t = (r / self.rc).clamp(0.0, 1.0)
        y = torch.zeros_like(t)
        if self.order == 1:
            y = y * t + 1.0
            y = y * t
        elif self.order == 2:
            y = y * t - 2.0
            y = y * t + 3.0
            y = y * t
            y = y * t
        elif self.order == 3:
            y = y * t + 6.0
            y = y * t - 15.0
            y = y * t + 10.0
            y = y * t
            y = y * t
            y = y * t
        elif self.order == 4:
            y = y * t - 20.0
            y = y * t + 70.0
            y = y * t - 84.0
            y = y * t + 35.0
            y = y * t
            y = y * t
            y = y * t
            y = y * t
        elif self.order == 5:
            y = y * t + 70.0
            y = y * t - 315.0
            y = y * t + 540.0
            y = y * t - 420.0
            y = y * t + 126.0
            y = y * t
            y = y * t
            y = y * t
            y = y * t
            y = y * t
        else:
            raise RuntimeError("smoothstep order must be in {1,2,3,4,5}")
        cutoff = (1.0 - y) * (r < self.rc).to(r.dtype)
        return cutoff.unsqueeze(-1)


class PainnMDFunctionalMessageLayer(torch.nn.Module):
    """PaiNN message layer compatible with torch.func transforms."""

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.natom_basis = int(base.natom_basis)
        self.radial = base.radial
        if hasattr(base.envelope, "order"):
            self.envelope = PainnMDFunctionalSmoothstepEnvelope(
                float(base.envelope.rc),
                int(base.envelope.order),
            )
        else:
            self.envelope = base.envelope
        self.interaction_context_network = base.interaction_context_network
        self.filter_network = base.filter_network

    def forward(
        self,
        node_scalar: Tensor,
        node_vector: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        context = self.interaction_context_network(node_scalar)
        distances = torch.norm(edge_weight, dim=-1)
        directions = edge_weight / distances.unsqueeze(-1)
        basis = self.radial(distances)
        cutoff = self.envelope(distances)
        filters = self.filter_network(basis) * cutoff
        index_i = edge_index[0]
        index_j = edge_index[1]
        messages = filters * context[index_j]
        scalar_message, radial_message, vector_message = torch.split(
            messages,
            self.natom_basis,
            dim=-1,
        )
        scalar_update = torch.zeros_like(node_scalar)
        scalar_update = torch.scatter_add(
            scalar_update,
            0,
            index_i.unsqueeze(1).expand_as(scalar_message),
            scalar_message,
        )
        directed_vector_message = (
            radial_message.unsqueeze(1) * directions[..., None]
            + vector_message.unsqueeze(1) * node_vector[index_j]
        )
        vector_update = torch.zeros_like(node_vector)
        vector_update = torch.scatter_add(
            vector_update,
            0,
            index_i.unsqueeze(-1).unsqueeze(-1).expand_as(directed_vector_message),
            directed_vector_message,
        )
        return node_scalar + scalar_update, node_vector + vector_update


class PainnMDEnergyModule(torch.nn.Module):
    """Pure directed-edge PaiNN energy graph for functional differentiation."""

    def __init__(self, base: torch.nn.Module, e0_lookup: Tensor):
        super().__init__()
        self.embedding = base.embedding
        self.message_layers = torch.nn.ModuleList(
            [PainnMDFunctionalMessageLayer(message) for message in base.message_layers]
        )
        self.mixing_layers = base.mixing_layers
        self.output = base.output
        self.register_buffer("e0_lookup", e0_lookup)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, Z: Tensor, edge_index: Tensor, edge_weight_e3: Tensor) -> Tensor:
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
        baseline = self.e0_lookup[Z].to(dtype=atom_energy.dtype).sum()
        return atom_energy.sum() + baseline


class PainnMDFunctionalInferenceWrapper(torch.nn.Module):
    """AOTAutograd-friendly energy/force wrapper using a pure energy graph."""

    def __init__(self, base: torch.nn.Module, e0_lookup: Tensor):
        super().__init__()
        self.energy_module = PainnMDEnergyModule(base, e0_lookup)

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
        edge_gradient, total_energy = torch.func.grad_and_value(
            self.energy_module,
            argnums=2,
        )(Z, edge_index, edge_weight_e3)

        index_i = edge_index[0]
        index_j = edge_index[1]
        atom_forces_i = torch.zeros(
            (Z.shape[0], 3),
            dtype=edge_gradient.dtype,
            device=edge_gradient.device,
        )
        atom_forces_j = torch.zeros_like(atom_forces_i)
        atom_forces_i = torch.scatter_add(
            atom_forces_i,
            0,
            index_i.unsqueeze(1).expand_as(edge_gradient),
            edge_gradient,
        )
        atom_forces_j = torch.scatter_add(
            atom_forces_j,
            0,
            index_j.unsqueeze(1).expand_as(edge_gradient),
            -edge_gradient,
        )
        atom_forces = atom_forces_i + atom_forces_j
        return total_energy, atom_forces.transpose(0, 1).contiguous()


class PainnMDPairedMessageLayer(torch.nn.Module):
    """Trained PaiNN message components used by the paired MD wrapper."""

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.natom_basis = int(base.natom_basis)
        self.interaction_context_network = base.interaction_context_network
        self.filter_network = base.filter_network


class PainnMDSharedGeometryInferenceWrapper(torch.nn.Module):
    """Directed-edge MD wrapper sharing geometry across interaction layers."""

    def __init__(
        self,
        base: torch.nn.Module,
        e0_lookup: Tensor,
        stacked_filters: bool = False,
    ):
        super().__init__()
        if len(base.message_layers) == 0:
            raise ValueError("shared-geometry PaiNN export requires message layers")
        self.embedding = base.embedding
        self.radial = base.message_layers[0].radial
        self.envelope = base.message_layers[0].envelope
        self.message_layers = torch.nn.ModuleList(
            [PainnMDPairedMessageLayer(message) for message in base.message_layers]
        )
        self.mixing_layers = base.mixing_layers
        self.output = base.output
        self.stacked_filters = bool(stacked_filters)
        self.num_interactions = len(base.message_layers)
        self.natom_basis = int(base.message_layers[0].natom_basis)
        self.register_buffer("e0_lookup", e0_lookup)

        filter_weights = []
        filter_biases = []
        for message in base.message_layers:
            linear = message.filter_network[0]
            if not isinstance(linear, torch.nn.Linear) or linear.bias is None:
                raise TypeError("stacked filter export requires biased Linear filters")
            filter_weights.append(linear.weight.detach().clone())
            filter_biases.append(linear.bias.detach().clone())
        self.register_buffer("stacked_filter_weight", torch.cat(filter_weights, dim=0))
        self.register_buffer("stacked_filter_bias", torch.cat(filter_biases, dim=0))
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
        distances = torch.norm(edge_weight_e3, dim=-1)
        directions = edge_weight_e3 / distances.unsqueeze(-1)
        basis = self.radial(distances)
        cutoff = self.envelope(distances)

        edge_count = edge_index.shape[1]
        if self.stacked_filters:
            stacked = F.linear(
                basis,
                self.stacked_filter_weight,
                self.stacked_filter_bias,
            ).reshape(
                edge_count,
                self.num_interactions,
                3 * self.natom_basis,
            )
        else:
            stacked = torch.empty(
                (edge_count, 0, 0),
                dtype=basis.dtype,
                device=basis.device,
            )

        index_i = edge_index[0]
        index_j = edge_index[1]
        node_scalar = self.embedding(Z)
        node_vector = torch.zeros(
            (node_scalar.shape[0], 3, node_scalar.shape[1]),
            dtype=node_scalar.dtype,
            device=node_scalar.device,
        )
        layer_index = 0
        for message, mixing in zip(self.message_layers, self.mixing_layers):
            context = message.interaction_context_network(node_scalar)
            if self.stacked_filters:
                filters = stacked[:, layer_index, :] * cutoff
            else:
                filters = message.filter_network(basis) * cutoff
            messages = filters * context[index_j]
            scalar_message, radial_message, vector_message = torch.split(
                messages,
                message.natom_basis,
                dim=-1,
            )

            scalar_update = torch.zeros_like(node_scalar)
            scalar_index = index_i.unsqueeze(1).expand_as(scalar_message)
            scalar_update = torch.scatter_add(
                scalar_update,
                0,
                scalar_index,
                scalar_message,
            )
            directed_vector_message = (
                radial_message.unsqueeze(1) * directions[..., None]
                + vector_message.unsqueeze(1) * node_vector[index_j]
            )
            vector_update = torch.zeros_like(node_vector)
            vector_index = index_i.unsqueeze(-1).unsqueeze(-1).expand_as(
                directed_vector_message
            )
            vector_update = torch.scatter_add(
                vector_update,
                0,
                vector_index,
                directed_vector_message,
            )
            node_scalar = node_scalar + scalar_update
            node_vector = node_vector + vector_update
            node_scalar, node_vector = mixing(node_scalar, node_vector)
            layer_index += 1

        atom_energy = self.output(node_scalar)
        edge_gradient = torch.autograd.grad(
            [atom_energy.sum()],
            [edge_weight_e3],
            create_graph=True,
        )[0]
        assert edge_gradient is not None
        edge_gradient = edge_gradient.detach()

        force_i = torch.zeros(
            (node_scalar.shape[0], 3),
            dtype=atom_energy.dtype,
            device=atom_energy.device,
        )
        force_j = torch.zeros_like(force_i)
        force_i = torch.scatter_add(
            force_i,
            0,
            index_i.unsqueeze(1).expand_as(edge_gradient),
            edge_gradient,
        )
        force_j = torch.scatter_add(
            force_j,
            0,
            index_j.unsqueeze(1).expand_as(edge_gradient),
            -edge_gradient,
        )
        forces_soa = (force_i + force_j).transpose(0, 1).contiguous()
        baseline = self.e0_lookup[Z].to(dtype=atom_energy.dtype).sum()
        return atom_energy.sum() + baseline, forces_soa


class PainnMDPairedInferenceWrapper(torch.nn.Module):
    """MD-only PaiNN wrapper that folds MD_CUDA reverse edges into pairs.

    MD_CUDA's ordinary NNP graph stores all forward pairs in the first half and
    their index-reversed, displacement-negated copies in the second half.  The
    wrapper evaluates geometry and radial filters only for the first half and
    reconstructs both directed PaiNN messages from each undirected pair.
    """

    def __init__(
        self,
        base: torch.nn.Module,
        e0_lookup: Tensor,
        stacked_filters: bool = False,
        return_pair_gradient: bool = False,
        validate_layout: bool = False,
        short_range_gate: torch.nn.Module | None = None,
    ):
        super().__init__()
        if len(base.message_layers) == 0:
            raise ValueError("paired PaiNN export requires at least one message layer")
        self.embedding = base.embedding
        self.radial = base.message_layers[0].radial
        self.envelope = base.message_layers[0].envelope
        self.message_layers = torch.nn.ModuleList(
            [PainnMDPairedMessageLayer(message) for message in base.message_layers]
        )
        self.mixing_layers = base.mixing_layers
        self.output = base.output
        self.stacked_filters = bool(stacked_filters)
        self.return_pair_gradient = bool(return_pair_gradient)
        self.validate_layout = bool(validate_layout)
        self.short_range_gate_enabled = short_range_gate is not None
        self.num_interactions = len(base.message_layers)
        self.natom_basis = int(base.message_layers[0].natom_basis)
        self.register_buffer("e0_lookup", e0_lookup)
        if short_range_gate is None:
            self.register_buffer(
                "short_range_gate_off_A", torch.zeros((1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "short_range_gate_full_A", torch.zeros((1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "short_range_gate_pair_enabled",
                torch.zeros((1, 1), dtype=torch.bool),
            )
        else:
            self.register_buffer(
                "short_range_gate_off_A",
                short_range_gate.off_A.detach().clone(),
            )
            self.register_buffer(
                "short_range_gate_full_A",
                short_range_gate.full_A.detach().clone(),
            )
            self.register_buffer(
                "short_range_gate_pair_enabled",
                short_range_gate.pair_enabled.detach().clone(),
            )

        filter_weights = []
        filter_biases = []
        for message in base.message_layers:
            linear = message.filter_network[0]
            if not isinstance(linear, torch.nn.Linear):
                raise TypeError("stacked filter export requires a Linear filter network")
            if linear.bias is None:
                raise ValueError("stacked filter export requires filter biases")
            filter_weights.append(linear.weight.detach().clone())
            filter_biases.append(linear.bias.detach().clone())
        self.register_buffer("stacked_filter_weight", torch.cat(filter_weights, dim=0))
        self.register_buffer("stacked_filter_bias", torch.cat(filter_biases, dim=0))

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
        edge_count = edge_index.shape[1]
        if edge_count % 2 != 0:
            raise RuntimeError("md_cuda_paired edge layout requires an even edge count")

        pair_count = edge_count // 2
        pair_index = edge_index[:, :pair_count]
        pair_weight = edge_weight[:, :pair_count]
        if self.validate_layout:
            reverse_index = edge_index[:, pair_count:]
            reverse_weight = edge_weight[:, pair_count:]
            if not torch.equal(reverse_index[0], pair_index[1]):
                raise RuntimeError("md_cuda_paired reverse sources do not match")
            if not torch.equal(reverse_index[1], pair_index[0]):
                raise RuntimeError("md_cuda_paired reverse destinations do not match")
            if not torch.equal(reverse_weight, -pair_weight):
                raise RuntimeError("md_cuda_paired reverse displacements do not match")
        pair_weight_e3 = pair_weight.transpose(0, 1).contiguous()
        pair_weight_e3 = pair_weight_e3.detach().requires_grad_(True)

        index_i = pair_index[0]
        index_j = pair_index[1]
        distances = torch.norm(pair_weight_e3, dim=-1)
        directions = pair_weight_e3 / distances.unsqueeze(-1)
        basis = self.radial(distances)
        cutoff = self.envelope(distances)
        pair_message_gate = torch.ones_like(distances)
        if self.short_range_gate_enabled:
            z_i = Z[index_i]
            z_j = Z[index_j]
            gate_enabled = self.short_range_gate_pair_enabled[z_i, z_j]
            gate_off = self.short_range_gate_off_A[z_i, z_j].to(
                dtype=distances.dtype
            )
            gate_full = self.short_range_gate_full_A[z_i, z_j].to(
                dtype=distances.dtype
            )
            gate_width = torch.where(
                gate_enabled,
                gate_full - gate_off,
                torch.ones_like(gate_full),
            )
            gate_t = ((distances - gate_off) / gate_width).clamp(0.0, 1.0)
            gate_smooth = gate_t.pow(3) * (
                10.0 - 15.0 * gate_t + 6.0 * gate_t.pow(2)
            )
            pair_message_gate = torch.where(
                gate_enabled,
                gate_smooth,
                pair_message_gate,
            )

        if self.stacked_filters:
            stacked = F.linear(
                basis,
                self.stacked_filter_weight,
                self.stacked_filter_bias,
            )
            stacked = stacked.reshape(
                pair_count,
                self.num_interactions,
                3 * self.natom_basis,
            )
        else:
            stacked = torch.empty(
                (pair_count, 0, 0),
                dtype=basis.dtype,
                device=basis.device,
            )

        node_scalar = self.embedding(Z)
        node_vector = torch.zeros(
            (node_scalar.shape[0], 3, node_scalar.shape[1]),
            dtype=node_scalar.dtype,
            device=node_scalar.device,
        )
        layer_index = 0
        for message, mixing in zip(self.message_layers, self.mixing_layers):
            context = message.interaction_context_network(node_scalar)
            if self.stacked_filters:
                filters = (
                    stacked[:, layer_index, :]
                    * cutoff
                    * pair_message_gate.unsqueeze(-1)
                )
            else:
                filters = (
                    message.filter_network(basis)
                    * cutoff
                    * pair_message_gate.unsqueeze(-1)
                )

            # Reconstruct the exact MD_CUDA directed-edge order (all forward
            # pairs followed by all reverse pairs).  Geometry and filter
            # projection stay shared, while using one scatter per update keeps
            # the numerical reduction path aligned with the directed wrapper.
            directed_target = torch.cat((index_i, index_j), dim=0)
            directed_source = torch.cat((index_j, index_i), dim=0)
            directed_filters = torch.cat((filters, filters), dim=0)
            directed_directions = torch.cat((directions, -directions), dim=0)
            messages = directed_filters * context[directed_source]
            scalar_message, radial_message, vector_message = torch.split(
                messages,
                message.natom_basis,
                dim=-1,
            )

            scalar_update = torch.zeros_like(node_scalar)
            scalar_index = directed_target.unsqueeze(1).expand_as(scalar_message)
            scalar_update = torch.scatter_add(
                scalar_update,
                0,
                scalar_index,
                scalar_message,
            )

            directed_vector_message = (
                radial_message.unsqueeze(1) * directed_directions[..., None]
                + vector_message.unsqueeze(1) * node_vector[directed_source]
            )
            vector_update = torch.zeros_like(node_vector)
            vector_index = (
                directed_target.unsqueeze(-1)
                .unsqueeze(-1)
                .expand_as(directed_vector_message)
            )
            vector_update = torch.scatter_add(
                vector_update,
                0,
                vector_index,
                directed_vector_message,
            )

            node_scalar = node_scalar + scalar_update
            node_vector = node_vector + vector_update
            node_scalar, node_vector = mixing(node_scalar, node_vector)
            layer_index += 1

        atom_energy = self.output(node_scalar)
        pair_gradient = torch.autograd.grad(
            [atom_energy.sum()],
            [pair_weight_e3],
            create_graph=True,
        )[0]
        assert pair_gradient is not None
        pair_gradient = pair_gradient.detach()

        baseline = self.e0_lookup[Z].to(dtype=atom_energy.dtype).sum()
        total_energy = atom_energy.sum() + baseline
        if self.return_pair_gradient:
            return total_energy, pair_gradient.transpose(0, 1).contiguous()

        atom_forces = torch.zeros(
            (node_scalar.shape[0], 3),
            dtype=atom_energy.dtype,
            device=atom_energy.device,
        )
        atom_forces = torch.index_add(atom_forces, 0, index_i, pair_gradient)
        atom_forces = torch.index_add(atom_forces, 0, index_j, -pair_gradient)
        return total_energy, atom_forces.transpose(0, 1).contiguous()


class RepulsiveMDKernel(torch.nn.Module):
    """TorchScript pair-energy kernel copied from a fixed SimpleGNN baseline."""

    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self.register_buffer("cutoff_A", source.cutoff_A.detach().clone())
        self.register_buffer("amplitude_eV", source.amplitude_eV.detach().clone())
        self.register_buffer(
            "zbl_screening_A", source.zbl_screening_A.detach().clone()
        )
        self.register_buffer("pair_enabled", source.pair_enabled.detach().clone())
        self.register_buffer(
            "zbl_coefficients", source.zbl_coefficients.detach().clone()
        )
        self.register_buffer("zbl_exponents", source.zbl_exponents.detach().clone())
        self.family_code = int(source.family_code)
        self.switch_start_fraction = float(source.switch_start_fraction)

    def forward(
        self, Z: Tensor, edge_index: Tensor, edge_weight_e3: Tensor
    ) -> Tensor:
        z_i = Z[edge_index[0]].long()
        z_j = Z[edge_index[1]].long()
        cutoff = self.cutoff_A[z_i, z_j].to(dtype=edge_weight_e3.dtype)
        amplitude = self.amplitude_eV[z_i, z_j].to(dtype=edge_weight_e3.dtype)
        screening = self.zbl_screening_A[z_i, z_j].to(
            dtype=edge_weight_e3.dtype
        )
        enabled = self.pair_enabled[z_i, z_j]
        z_product = (z_i * z_j).to(dtype=edge_weight_e3.dtype)
        distance = torch.linalg.vector_norm(edge_weight_e3, dim=-1).clamp_min(
            1.0e-8
        )
        inside = enabled & (distance < cutoff)
        safe_cutoff = torch.where(enabled, cutoff, torch.ones_like(cutoff))

        if self.family_code == 0:
            t = distance / safe_cutoff
            raw = amplitude * torch.pow(t, -6.0) * torch.pow(1.0 - t, 4.0)
            return torch.where(inside, raw, torch.zeros_like(raw))

        safe_screening = torch.where(
            enabled, screening, torch.ones_like(screening)
        )
        x = distance / safe_screening
        phi = torch.zeros_like(distance)
        coefficients = self.zbl_coefficients.to(dtype=edge_weight_e3.dtype)
        exponents = self.zbl_exponents.to(dtype=edge_weight_e3.dtype)
        for index in range(4):
            phi = phi + coefficients[index] * torch.exp(-exponents[index] * x)
        zbl = 14.3996454784255 * z_product * phi / distance
        switch_start = self.switch_start_fraction * safe_cutoff
        s = ((distance - switch_start) / (safe_cutoff - switch_start)).clamp(
            0.0, 1.0
        )
        switch = (
            1.0
            - 10.0 * torch.pow(s, 3.0)
            + 15.0 * torch.pow(s, 4.0)
            - 6.0 * torch.pow(s, 5.0)
        )
        return torch.where(inside, zbl * switch, torch.zeros_like(zbl))


class PainnMDRepulsiveWrapper(torch.nn.Module):
    """Add the fixed pair term to any three-input MD inference wrapper."""

    def __init__(
        self,
        base: torch.nn.Module,
        repulsive: RepulsiveMDKernel,
        paired: bool = False,
        return_pair_gradient: bool = False,
        base_output_is_pair_gradient: bool = False,
    ):
        super().__init__()
        self.base = base
        self.repulsive = repulsive
        self.paired = bool(paired)
        self.return_pair_gradient = bool(return_pair_gradient)
        self.base_output_is_pair_gradient = bool(base_output_is_pair_gradient)

    def forward(
        self, Z: Tensor, edge_index: Tensor, edge_weight: Tensor
    ) -> Tuple[Tensor, Tensor]:
        base_energy, base_output = self.base(Z, edge_index, edge_weight)
        if Z.dtype != torch.long:
            Z = Z.long()
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        selected_index = edge_index
        selected_weight = edge_weight
        multiplier = 0.5
        if self.paired:
            edge_count = edge_index.shape[1]
            if edge_count % 2 != 0:
                raise RuntimeError(
                    "md_cuda_paired repulsive layout requires an even edge count"
                )
            pair_count = edge_count // 2
            selected_index = edge_index[:, :pair_count]
            selected_weight = edge_weight[:, :pair_count]
            multiplier = 1.0

        selected_weight_e3 = selected_weight.transpose(0, 1).contiguous()
        selected_weight_e3 = selected_weight_e3.detach().requires_grad_(True)
        edge_energy = self.repulsive(Z, selected_index, selected_weight_e3)
        repulsive_energy = multiplier * edge_energy.sum()
        gradient = torch.autograd.grad(
            [repulsive_energy], [selected_weight_e3], create_graph=True
        )[0]
        assert gradient is not None
        gradient = gradient.detach()

        if self.return_pair_gradient:
            if not self.base_output_is_pair_gradient:
                raise RuntimeError(
                    "pair-gradient output requires a pair-gradient base wrapper"
                )
            return (
                base_energy + repulsive_energy,
                base_output + gradient.transpose(0, 1).contiguous(),
            )

        if self.paired and self.base_output_is_pair_gradient:
            combined_gradient = (
                base_output + gradient.transpose(0, 1).contiguous()
            ).transpose(0, 1)
            atom_forces = torch.zeros(
                (Z.shape[0], 3),
                dtype=combined_gradient.dtype,
                device=combined_gradient.device,
            )
            atom_forces = torch.index_add(
                atom_forces, 0, selected_index[0], combined_gradient
            )
            atom_forces = torch.index_add(
                atom_forces, 0, selected_index[1], -combined_gradient
            )
            return (
                base_energy + repulsive_energy,
                atom_forces.transpose(0, 1).contiguous(),
            )

        force_i = torch.zeros(
            (Z.shape[0], 3), dtype=gradient.dtype, device=gradient.device
        )
        force_j = torch.zeros_like(force_i)
        force_i = torch.index_add(
            force_i, 0, selected_index[0], gradient
        )
        force_j = torch.index_add(
            force_j, 0, selected_index[1], -gradient
        )
        repulsive_forces = (force_i + force_j).transpose(0, 1).contiguous()
        return base_energy + repulsive_energy, base_output + repulsive_forces


class PainnMDCSRRepulsiveWrapper(torch.nn.Module):
    """Add the fixed pair term while preserving the four-input CSR ABI."""

    def __init__(self, base: torch.nn.Module, repulsive: RepulsiveMDKernel):
        super().__init__()
        self.base = base
        self.repulsive = repulsive

    def forward(
        self,
        Z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        offsets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        base_energy, base_forces = self.base(
            Z, edge_index, edge_weight, offsets
        )
        if Z.dtype != torch.long:
            Z = Z.long()
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        edge_weight_e3 = edge_weight.transpose(0, 1).contiguous()
        edge_weight_e3 = edge_weight_e3.detach().requires_grad_(True)
        edge_energy = self.repulsive(Z, edge_index, edge_weight_e3)
        repulsive_energy = 0.5 * edge_energy.sum()
        gradient = torch.autograd.grad(
            [repulsive_energy], [edge_weight_e3], create_graph=True
        )[0]
        assert gradient is not None
        gradient = gradient.detach()
        force_i = torch.zeros(
            (Z.shape[0], 3), dtype=gradient.dtype, device=gradient.device
        )
        force_j = torch.zeros_like(force_i)
        force_i = torch.index_add(force_i, 0, edge_index[0], gradient)
        force_j = torch.index_add(force_j, 0, edge_index[1], -gradient)
        repulsive_forces = (force_i + force_j).transpose(0, 1).contiguous()
        return (
            base_energy + repulsive_energy,
            base_forces + repulsive_forces,
        )


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


def _validate_shared_geometry(base: torch.nn.Module) -> None:
    """Require every interaction layer to use identical geometry modules."""

    if len(base.message_layers) == 0:
        raise ValueError("shared geometry requires at least one message layer")
    reference = base.message_layers[0]
    for layer_index, message in enumerate(base.message_layers[1:], start=1):
        if type(message.radial) is not type(reference.radial):
            raise ValueError(f"message layer {layer_index} has a different radial type")
        if type(message.envelope) is not type(reference.envelope):
            raise ValueError(f"message layer {layer_index} has a different envelope type")
        if float(message.cutoff) != float(reference.cutoff):
            raise ValueError(f"message layer {layer_index} has a different cutoff")
        reference_radial = reference.radial.state_dict()
        candidate_radial = message.radial.state_dict()
        if reference_radial.keys() != candidate_radial.keys():
            raise ValueError(f"message layer {layer_index} has different radial buffers")
        for name, value in reference_radial.items():
            if not torch.equal(value, candidate_radial[name]):
                raise ValueError(
                    f"message layer {layer_index} radial buffer {name!r} differs"
                )
        reference_envelope = reference.envelope.state_dict()
        candidate_envelope = message.envelope.state_dict()
        if reference_envelope.keys() != candidate_envelope.keys():
            raise ValueError(f"message layer {layer_index} has different envelope buffers")
        for name, value in reference_envelope.items():
            if not torch.equal(value, candidate_envelope[name]):
                raise ValueError(
                    f"message layer {layer_index} envelope buffer {name!r} differs"
                )


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
        "--repulsive-baseline-json",
        type=Path,
        default=None,
        help="Fixed repulsive pair-potential JSON composed into every MD path.",
    )
    parser.add_argument(
        "--short-range-gate-json",
        type=Path,
        default=None,
        help=(
            "Fixed pair-message gate JSON for inference-only safety. "
            "Currently supported by the paired/shared MD_CUDA path."
        ),
    )
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
    parser.add_argument(
        "--edge-layout",
        choices=("directed", "md_cuda_paired"),
        default="directed",
        help="Use ordinary directed edges or fold MD_CUDA's paired reverse-edge layout.",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=("per_layer", "shared"),
        default="per_layer",
        help="Evaluate geometry per message layer or share it across paired message layers.",
    )
    parser.add_argument(
        "--filter-projection",
        choices=("per_layer", "stacked"),
        default="per_layer",
        help="Evaluate filter linears per layer or as one stacked projection.",
    )
    parser.add_argument(
        "--force-output",
        choices=("atom", "pair_gradient"),
        default="atom",
    )
    parser.add_argument(
        "--validate-pair-layout",
        action="store_true",
        help="Validate reverse indices/displacements on every paired forward (debug only).",
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
    from simplegnn.repulsive import RepulsivePairPotential
    from simplegnn.short_range_gate import ShortRangePairMessageGate

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
    if args.edge_layout == "md_cuda_paired" and args.forward_mode != "md":
        raise ValueError("--edge-layout md_cuda_paired requires --forward-mode md")
    if args.edge_layout == "md_cuda_paired" and args.aggregation_mode != "scatter":
        raise ValueError("--edge-layout md_cuda_paired is incompatible with CSR aggregation")
    if args.edge_layout == "md_cuda_paired" and args.geometry_mode != "shared":
        raise ValueError("--edge-layout md_cuda_paired requires --geometry-mode shared")
    if args.filter_projection == "stacked" and args.geometry_mode != "shared":
        raise ValueError("--filter-projection stacked requires --geometry-mode shared")
    if args.force_output == "pair_gradient" and args.edge_layout != "md_cuda_paired":
        raise ValueError("--force-output pair_gradient requires --edge-layout md_cuda_paired")
    if (
        args.short_range_gate_json is not None
        and args.edge_layout != "md_cuda_paired"
    ):
        raise ValueError(
            "--short-range-gate-json currently requires "
            "--edge-layout md_cuda_paired"
        )
    short_range_gate = None
    if args.short_range_gate_json is not None:
        short_range_gate = ShortRangePairMessageGate.from_json(
            args.short_range_gate_json
        )
    if args.aggregation_mode == "csr":
        wrapper = PainnMDCSRInferenceWrapper(base, e0_lookup).to(device).eval()
    elif args.edge_layout == "md_cuda_paired":
        _validate_shared_geometry(base)
        wrapper = PainnMDPairedInferenceWrapper(
            base,
            e0_lookup,
            stacked_filters=args.filter_projection == "stacked",
            return_pair_gradient=(
                args.force_output == "pair_gradient"
                or args.repulsive_baseline_json is not None
            ),
            validate_layout=args.validate_pair_layout,
            short_range_gate=short_range_gate,
        ).to(device).eval()
    elif args.geometry_mode == "shared":
        _validate_shared_geometry(base)
        wrapper = PainnMDSharedGeometryInferenceWrapper(
            base,
            e0_lookup,
            stacked_filters=args.filter_projection == "stacked",
        ).to(device).eval()
    elif args.forward_mode == "md":
        wrapper = PainnMDInferenceWrapper(base, e0_lookup).to(device).eval()
    else:
        wrapper = PainnMDNNPWrapper(base, e0_lookup).to(device).eval()
    if args.repulsive_baseline_json is not None:
        repulsive_source = RepulsivePairPotential.from_json(
            args.repulsive_baseline_json
        )
        repulsive_kernel = RepulsiveMDKernel(repulsive_source)
        if args.aggregation_mode == "csr":
            wrapper = PainnMDCSRRepulsiveWrapper(
                wrapper, repulsive_kernel
            ).to(device).eval()
        else:
            wrapper = PainnMDRepulsiveWrapper(
                wrapper,
                repulsive_kernel,
                paired=args.edge_layout == "md_cuda_paired",
                return_pair_gradient=args.force_output == "pair_gradient",
                base_output_is_pair_gradient=(
                    args.edge_layout == "md_cuda_paired"
                ),
            ).to(device).eval()
    scripted = torch.jit.script(wrapper)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
