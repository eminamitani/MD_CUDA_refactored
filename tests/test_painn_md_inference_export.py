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

    sys.path.insert(0, str(ROOT / "scripts"))
    from export_simplegnn_painn_for_md import PainnMDInferenceWrapper, PainnMDNNPWrapper
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


if __name__ == "__main__":
    unittest.main()
