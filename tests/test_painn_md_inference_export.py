from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIMPLEGNN_ROOT = Path(os.environ.get("SIMPLEGNN_ROOT", ROOT.parent / "simplegnn_version2"))
TORCH_SPEC = importlib.util.find_spec("torch")

if TORCH_SPEC is not None:
    import torch

    sys.path.insert(0, str(SIMPLEGNN_ROOT))
    from simplegnn.painn import Painn
    from simplegnn.repulsive import RepulsivePairPotential
    from simplegnn.short_range_gate import (
        PainnWithInferenceGate,
        ShortRangePairMessageGate,
    )

    sys.path.insert(0, str(ROOT / "scripts"))
    from export_simplegnn_painn_for_md import (
        PainnMDFunctionalInferenceWrapper,
        PainnMDCSRInferenceWrapper,
        PainnMDInferenceWrapper,
        PainnMDNNPWrapper,
        PainnMDPairedInferenceWrapper,
        PainnMDRepulsiveWrapper,
        PainnMDCSRRepulsiveWrapper,
        PainnMDSharedGeometryInferenceWrapper,
        RepulsiveMDKernel,
        _validate_shared_geometry,
    )
else:  # pragma: no cover - depends on the optional local ML runtime.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not available")
class PainnMDInferenceExportTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.base = Painn(
            natom_basis=8,
            n_radial=6,
            cutoff=5.0,
            epsilon=1e-7,
            num_interactions=2,
            radial_type="gauss",
            envelope_type="smoothstep",
        ).eval()
        self.atomic_numbers = torch.tensor([3, 8, 14], dtype=torch.long)
        self.edge_index = torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
            dtype=torch.long,
        )
        self.edge_weight = torch.tensor(
            [
                [1.2, -1.2, 0.3, -0.3, -0.9, 0.9],
                [0.1, -0.1, 1.1, -1.1, 0.7, -0.7],
                [0.2, -0.2, -0.4, 0.4, 0.6, -0.6],
            ],
            dtype=torch.float32,
        )
        self.baseline = torch.zeros(119, dtype=torch.float32)
        self.baseline[3] = 1.5
        self.baseline[8] = -0.5
        self.repulsive = RepulsivePairPotential(
            {
                "schema_version": 1,
                "family": "inverse_power",
                "anchor_fraction": 0.8,
                "anchor_force_eV_per_A": 50.0,
                "pairs": {"3-8": {"cutoff_A": 1.65}},
            }
        ).eval()

    def csr_graph(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = torch.argsort(self.edge_index[0], stable=True)
        edge_index = self.edge_index[:, order]
        edge_weight = self.edge_weight[:, order]
        counts = torch.bincount(edge_index[0], minlength=self.atomic_numbers.shape[0])
        offsets = torch.zeros(self.atomic_numbers.shape[0] + 1, dtype=torch.long)
        offsets[1:] = torch.cumsum(counts, dim=0)
        return edge_index, edge_weight, offsets

    def paired_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        forward = torch.tensor([0, 2, 4], dtype=torch.long)
        reverse = torch.tensor([1, 3, 5], dtype=torch.long)
        order = torch.cat((forward, reverse))
        return self.edge_index[:, order], self.edge_weight[:, order]

    def test_md_forward_matches_legacy_energy_and_forces(self) -> None:
        legacy = PainnMDNNPWrapper(self.base, self.baseline).eval()
        optimized = PainnMDInferenceWrapper(self.base, self.baseline).eval()

        legacy_energy, legacy_forces = legacy(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight,
        )
        optimized_energy, optimized_forces = optimized(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight,
        )

        self.assertLessEqual(float(torch.abs(legacy_energy - optimized_energy)), 1e-4)
        self.assertLessEqual(float(torch.max(torch.abs(legacy_forces - optimized_forces))), 1e-5)

    def test_md_forward_is_torchscript_compatible(self) -> None:
        optimized = PainnMDInferenceWrapper(self.base, self.baseline).eval()
        scripted = torch.jit.script(optimized)
        energy, forces = scripted(self.atomic_numbers, self.edge_index, self.edge_weight)

        self.assertEqual(tuple(energy.shape), ())
        self.assertEqual(tuple(forces.shape), (3, 3))
        self.assertTrue(torch.isfinite(energy))
        self.assertTrue(torch.isfinite(forces).all())

    def test_functional_forward_matches_md_forward(self) -> None:
        directed = PainnMDInferenceWrapper(self.base, self.baseline).eval()
        functional = PainnMDFunctionalInferenceWrapper(self.base, self.baseline).eval()
        directed_energy, directed_forces = directed(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight,
        )
        functional_energy, functional_forces = functional(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight,
        )
        self.assertLessEqual(float(torch.abs(directed_energy - functional_energy)), 1e-4)
        self.assertLessEqual(
            float(torch.max(torch.abs(directed_forces - functional_forces))),
            1e-5,
        )

    def test_paired_shared_matches_directed_energy_and_forces(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        directed = PainnMDInferenceWrapper(self.base, self.baseline).eval()
        paired = PainnMDPairedInferenceWrapper(
            self.base,
            self.baseline,
            validate_layout=True,
        ).eval()

        directed_energy, directed_forces = directed(
            self.atomic_numbers,
            edge_index,
            edge_weight,
        )
        paired_energy, paired_forces = paired(
            self.atomic_numbers,
            edge_index,
            edge_weight,
        )

        self.assertLessEqual(float(torch.abs(directed_energy - paired_energy)), 1e-4)
        self.assertLessEqual(float(torch.max(torch.abs(directed_forces - paired_forces))), 1e-5)

    def test_paired_short_range_gate_matches_simplegnn_gate(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        gate = ShortRangePairMessageGate(
            {
                "schema_version": 1,
                "kind": "pair_message_gate",
                "transition": "quintic_smoothstep",
                "pairs": {
                    "3-8": {"off_A": 1.1, "full_A": 1.4},
                    "3-14": {"off_A": 0.8, "full_A": 1.6},
                    "8-14": {"off_A": 0.7, "full_A": 1.3},
                },
            }
        ).eval()
        simplegnn_gate = PainnWithInferenceGate(self.base, gate).eval()
        reference_energy, reference_forces, _ = simplegnn_gate(
            self.atomic_numbers,
            edge_index,
            edge_weight.transpose(0, 1).contiguous(),
            batch=None,
        )
        paired = PainnMDPairedInferenceWrapper(
            self.base,
            torch.zeros_like(self.baseline),
            short_range_gate=gate,
            validate_layout=True,
        ).eval()
        paired_energy, paired_forces = paired(
            self.atomic_numbers,
            edge_index,
            edge_weight,
        )
        self.assertLessEqual(
            float(torch.abs(reference_energy - paired_energy)), 1e-4
        )
        self.assertLessEqual(
            float(
                torch.max(
                    torch.abs(
                        reference_forces.transpose(0, 1) - paired_forces
                    )
                )
            ),
            1e-5,
        )
        scripted = torch.jit.script(paired)
        script_energy, script_forces = scripted(
            self.atomic_numbers, edge_index, edge_weight
        )
        self.assertLessEqual(
            float(torch.abs(paired_energy - script_energy)), 1e-4
        )
        self.assertLessEqual(
            float(torch.max(torch.abs(paired_forces - script_forces))), 1e-5
        )

    def test_directed_shared_geometry_matches_md_forward(self) -> None:
        directed = PainnMDInferenceWrapper(self.base, self.baseline).eval()
        shared = PainnMDSharedGeometryInferenceWrapper(self.base, self.baseline).eval()

        directed_energy, directed_forces = directed(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight,
        )
        shared_energy, shared_forces = shared(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight,
        )

        self.assertLessEqual(float(torch.abs(directed_energy - shared_energy)), 1e-4)
        self.assertLessEqual(float(torch.max(torch.abs(directed_forces - shared_forces))), 1e-5)

    def test_directed_shared_stacked_matches_per_layer(self) -> None:
        shared = PainnMDSharedGeometryInferenceWrapper(self.base, self.baseline).eval()
        stacked = PainnMDSharedGeometryInferenceWrapper(
            self.base,
            self.baseline,
            stacked_filters=True,
        ).eval()
        shared_energy, shared_forces = shared(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight,
        )
        stacked_energy, stacked_forces = stacked(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight,
        )
        self.assertLessEqual(float(torch.abs(shared_energy - stacked_energy)), 1e-4)
        self.assertLessEqual(float(torch.max(torch.abs(shared_forces - stacked_forces))), 1e-5)

    def test_paired_stacked_filters_match_per_layer(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        per_layer = PainnMDPairedInferenceWrapper(
            self.base,
            self.baseline,
        ).eval()
        stacked = PainnMDPairedInferenceWrapper(
            self.base,
            self.baseline,
            stacked_filters=True,
        ).eval()

        reference_energy, reference_forces = per_layer(
            self.atomic_numbers,
            edge_index,
            edge_weight,
        )
        stacked_energy, stacked_forces = stacked(
            self.atomic_numbers,
            edge_index,
            edge_weight,
        )

        self.assertLessEqual(float(torch.abs(reference_energy - stacked_energy)), 1e-4)
        self.assertLessEqual(float(torch.max(torch.abs(reference_forces - stacked_forces))), 1e-5)

    def test_paired_pair_gradient_reconstructs_atom_forces(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        atom_wrapper = PainnMDPairedInferenceWrapper(
            self.base,
            self.baseline,
        ).eval()
        gradient_wrapper = PainnMDPairedInferenceWrapper(
            self.base,
            self.baseline,
            return_pair_gradient=True,
        ).eval()

        atom_energy, atom_forces = atom_wrapper(
            self.atomic_numbers,
            edge_index,
            edge_weight,
        )
        gradient_energy, pair_gradient = gradient_wrapper(
            self.atomic_numbers,
            edge_index,
            edge_weight,
        )
        pair_index = edge_index[:, : edge_index.shape[1] // 2]
        reconstructed = torch.zeros((self.atomic_numbers.shape[0], 3))
        reconstructed.index_add_(0, pair_index[0], pair_gradient.transpose(0, 1))
        reconstructed.index_add_(0, pair_index[1], -pair_gradient.transpose(0, 1))

        self.assertLessEqual(float(torch.abs(atom_energy - gradient_energy)), 1e-4)
        self.assertLessEqual(
            float(torch.max(torch.abs(atom_forces - reconstructed.transpose(0, 1)))),
            1e-5,
        )

    def test_paired_forward_is_torchscript_compatible(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        paired = PainnMDPairedInferenceWrapper(self.base, self.baseline).eval()
        scripted = torch.jit.script(paired)
        energy, forces = scripted(self.atomic_numbers, edge_index, edge_weight)

        self.assertEqual(tuple(energy.shape), ())
        self.assertEqual(tuple(forces.shape), (3, 3))
        self.assertTrue(torch.isfinite(energy))
        self.assertTrue(torch.isfinite(forces).all())

    def test_paired_layout_validation_rejects_malformed_reverse_edges(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        edge_weight = edge_weight.clone()
        edge_weight[0, -1] += 0.25
        paired = PainnMDPairedInferenceWrapper(
            self.base,
            self.baseline,
            validate_layout=True,
        ).eval()
        with self.assertRaisesRegex(RuntimeError, "reverse displacements"):
            paired(self.atomic_numbers, edge_index, edge_weight)

    def test_paired_layout_rejects_odd_edge_count(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        paired = PainnMDPairedInferenceWrapper(self.base, self.baseline).eval()
        with self.assertRaisesRegex(RuntimeError, "even edge count"):
            paired(self.atomic_numbers, edge_index[:, :-1], edge_weight[:, :-1])

    def test_shared_geometry_validation_rejects_different_radial_buffer(self) -> None:
        with torch.no_grad():
            self.base.message_layers[1].radial.offsets[0] += 0.1
        with self.assertRaisesRegex(ValueError, "radial buffer"):
            _validate_shared_geometry(self.base)

    def test_csr_forward_matches_scatter_energy_and_forces(self) -> None:
        edge_index, edge_weight, offsets = self.csr_graph()
        scatter = PainnMDInferenceWrapper(self.base, self.baseline).eval()
        csr = PainnMDCSRInferenceWrapper(self.base, self.baseline).eval()

        scatter_energy, scatter_forces = scatter(
            self.atomic_numbers,
            edge_index,
            edge_weight,
        )
        csr_energy, csr_forces = csr(
            self.atomic_numbers,
            edge_index,
            edge_weight,
            offsets,
        )

        self.assertLessEqual(float(torch.abs(scatter_energy - csr_energy)), 1e-4)
        self.assertLessEqual(float(torch.max(torch.abs(scatter_forces - csr_forces))), 1e-5)

    def test_csr_forward_is_torchscript_compatible(self) -> None:
        edge_index, edge_weight, offsets = self.csr_graph()
        csr = PainnMDCSRInferenceWrapper(self.base, self.baseline).eval()
        scripted = torch.jit.script(csr)
        energy, forces = scripted(
            self.atomic_numbers,
            edge_index,
            edge_weight,
            offsets,
        )

        self.assertEqual(tuple(energy.shape), ())
        self.assertEqual(tuple(forces.shape), (3, 3))
        self.assertTrue(torch.isfinite(energy))
        self.assertTrue(torch.isfinite(forces).all())

    def test_csr_forward_supports_zero_neighbour_segments(self) -> None:
        atomic_numbers = torch.tensor([3, 8, 14, 8], dtype=torch.long)
        edge_index = torch.tensor([[0, 0, 2], [1, 2, 0]], dtype=torch.long)
        edge_weight = torch.tensor(
            [[1.2, 0.3, -0.3], [0.1, 1.1, -1.1], [0.2, -0.4, 0.4]],
            dtype=torch.float32,
        )
        offsets = torch.tensor([0, 2, 2, 3, 3], dtype=torch.long)
        scatter = PainnMDInferenceWrapper(self.base, self.baseline).eval()
        csr = PainnMDCSRInferenceWrapper(self.base, self.baseline).eval()

        scatter_energy, scatter_forces = scatter(atomic_numbers, edge_index, edge_weight)
        csr_energy, csr_forces = csr(atomic_numbers, edge_index, edge_weight, offsets)

        self.assertLessEqual(float(torch.abs(scatter_energy - csr_energy)), 1e-4)
        self.assertLessEqual(float(torch.max(torch.abs(scatter_forces - csr_forces))), 1e-5)

    def test_distributed_padding_has_zero_contribution(self) -> None:
        optimized = PainnMDInferenceWrapper(self.base, self.baseline).eval()
        reference_energy, reference_forces = optimized(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight,
        )

        padding_count = 7
        padding_nodes = torch.arange(padding_count, dtype=torch.long) % self.atomic_numbers.shape[0]
        padding_edge_index = torch.stack((padding_nodes, padding_nodes), dim=0)
        padding_edge_weight = torch.zeros((3, padding_count), dtype=torch.float32)
        padding_edge_weight[0] = 1.0e5
        padded_energy, padded_forces = optimized(
            self.atomic_numbers,
            torch.cat((self.edge_index, padding_edge_index), dim=1),
            torch.cat((self.edge_weight, padding_edge_weight), dim=1),
        )

        self.assertLessEqual(float(torch.abs(reference_energy - padded_energy)), 1e-6)
        self.assertLessEqual(float(torch.max(torch.abs(reference_forces - padded_forces))), 1e-6)

    def test_repulsive_directed_wrapper_matches_simplegnn_total(self) -> None:
        base_wrapper = PainnMDInferenceWrapper(self.base, self.baseline).eval()
        wrapped = PainnMDRepulsiveWrapper(
            base_wrapper, RepulsiveMDKernel(self.repulsive)
        ).eval()
        base_energy, base_forces = base_wrapper(
            self.atomic_numbers, self.edge_index, self.edge_weight
        )
        rep_energy, rep_forces, _ = self.repulsive(
            self.atomic_numbers,
            self.edge_index,
            self.edge_weight.transpose(0, 1),
            batch=None,
        )
        total_energy, total_forces = wrapped(
            self.atomic_numbers, self.edge_index, self.edge_weight
        )
        self.assertLessEqual(
            float(torch.abs(total_energy - base_energy - rep_energy)), 1e-6
        )
        self.assertLessEqual(
            float(
                torch.max(
                    torch.abs(
                        total_forces - base_forces - rep_forces.transpose(0, 1)
                    )
                )
            ),
            1e-6,
        )

    def test_repulsive_paired_and_directed_wrappers_match(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        directed = PainnMDRepulsiveWrapper(
            PainnMDInferenceWrapper(self.base, self.baseline).eval(),
            RepulsiveMDKernel(self.repulsive),
        ).eval()
        paired = PainnMDRepulsiveWrapper(
            PainnMDPairedInferenceWrapper(self.base, self.baseline).eval(),
            RepulsiveMDKernel(self.repulsive),
            paired=True,
        ).eval()
        directed_energy, directed_forces = directed(
            self.atomic_numbers, edge_index, edge_weight
        )
        paired_energy, paired_forces = paired(
            self.atomic_numbers, edge_index, edge_weight
        )
        self.assertLessEqual(
            float(torch.abs(directed_energy - paired_energy)), 1e-4
        )
        self.assertLessEqual(
            float(torch.max(torch.abs(directed_forces - paired_forces))), 1e-5
        )

    def test_repulsive_pair_gradient_reconstructs_atom_forces(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        atom_wrapper = PainnMDRepulsiveWrapper(
            PainnMDPairedInferenceWrapper(
                self.base, self.baseline, return_pair_gradient=True
            ).eval(),
            RepulsiveMDKernel(self.repulsive),
            paired=True,
            base_output_is_pair_gradient=True,
        ).eval()
        gradient_wrapper = PainnMDRepulsiveWrapper(
            PainnMDPairedInferenceWrapper(
                self.base, self.baseline, return_pair_gradient=True
            ).eval(),
            RepulsiveMDKernel(self.repulsive),
            paired=True,
            return_pair_gradient=True,
            base_output_is_pair_gradient=True,
        ).eval()
        atom_energy, atom_forces = atom_wrapper(
            self.atomic_numbers, edge_index, edge_weight
        )
        gradient_energy, pair_gradient = gradient_wrapper(
            self.atomic_numbers, edge_index, edge_weight
        )
        pair_index = edge_index[:, : edge_index.shape[1] // 2]
        reconstructed = torch.zeros(
            (self.atomic_numbers.shape[0], 3), dtype=pair_gradient.dtype
        )
        reconstructed.index_add_(
            0, pair_index[0], pair_gradient.transpose(0, 1)
        )
        reconstructed.index_add_(
            0, pair_index[1], -pair_gradient.transpose(0, 1)
        )
        self.assertLessEqual(float(torch.abs(atom_energy - gradient_energy)), 1e-4)
        self.assertLessEqual(
            float(
                torch.max(
                    torch.abs(atom_forces - reconstructed.transpose(0, 1))
                )
            ),
            1e-5,
        )

    def test_repulsive_csr_matches_scatter_and_scripts(self) -> None:
        edge_index, edge_weight, offsets = self.csr_graph()
        scatter = PainnMDRepulsiveWrapper(
            PainnMDInferenceWrapper(self.base, self.baseline).eval(),
            RepulsiveMDKernel(self.repulsive),
        ).eval()
        csr = PainnMDCSRRepulsiveWrapper(
            PainnMDCSRInferenceWrapper(self.base, self.baseline).eval(),
            RepulsiveMDKernel(self.repulsive),
        ).eval()
        scatter_energy, scatter_forces = scatter(
            self.atomic_numbers, edge_index, edge_weight
        )
        scripted = torch.jit.script(csr)
        csr_energy, csr_forces = scripted(
            self.atomic_numbers, edge_index, edge_weight, offsets
        )
        self.assertLessEqual(float(torch.abs(scatter_energy - csr_energy)), 1e-4)
        self.assertLessEqual(
            float(torch.max(torch.abs(scatter_forces - csr_forces))), 1e-5
        )

    def test_repulsive_paired_wrapper_is_torchscript_compatible(self) -> None:
        edge_index, edge_weight = self.paired_graph()
        wrapped = PainnMDRepulsiveWrapper(
            PainnMDPairedInferenceWrapper(self.base, self.baseline).eval(),
            RepulsiveMDKernel(self.repulsive),
            paired=True,
        ).eval()
        scripted = torch.jit.script(wrapped)
        energy, forces = scripted(
            self.atomic_numbers, edge_index, edge_weight
        )
        self.assertTrue(torch.isfinite(energy))
        self.assertTrue(torch.isfinite(forces).all())


if __name__ == "__main__":
    unittest.main()
