import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GENERATOR = load_script("generate_painn_pair_md_diagnostic_config.py")
COMPARATOR = load_script("compare_painn_pair_md_trajectories.py")
ASSESSOR = load_script("assess_painn_pair_md_propagation.py")


class PairMDDiagnosticTests(unittest.TestCase):
    def test_generator_selects_full_stepwise_trajectory(self):
        args = argparse.Namespace(
            name="test",
            input_xyz=Path("input.xyz"),
            model=Path("model.pt"),
            trajectory=Path("trajectory.xyz"),
            ensemble="NVT",
            steps=100,
            dt_fs=0.5,
            temperature_k=1500.0,
            seed=17,
            max_edges=500000,
        )
        config = GENERATOR.build_config(args)
        observer = config["steps"][0]["observer"]
        self.assertEqual(observer["interval"], 1)
        self.assertEqual(observer["trajectory"]["fields"], ["position", "velocity", "force", "energy"])
        self.assertTrue(config["steps"][0]["simulation"]["ensemble"]["rescale_initial_temperature"])

    def test_comparator_reports_float32_aware_force_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.xyz"
            candidate = root / "candidate.xyz"
            template = (
                "1\n"
                'Properties=species:S:1:pos:R:3:velocities:R:3:forces:R:3 energy={energy} time_fs=0\n'
                "Li 0 0 0 0 0 0 {force} 0 0\n"
            )
            reference.write_text(template.format(energy="1.0", force="1.0"))
            candidate.write_text(template.format(energy="1.00001", force="1.00003"))
            args = argparse.Namespace(
                reference=reference,
                candidate=candidate,
                energy_max_ev=1.0e-4,
                position_max_angstrom=1.0e-4,
                velocity_max_angstrom_per_fs=1.0e-5,
                force_max_ev_per_angstrom=5.0e-5,
                force_rms_ev_per_angstrom=2.0e-5,
            )
            result = COMPARATOR.compare(args)
            self.assertEqual(result["status"], "pass")
            self.assertAlmostEqual(result["maxima"]["force_max_abs_ev_per_angstrom"], 3.0e-5)
            self.assertAlmostEqual(result["maxima"]["force_relative_l2"], 3.0e-5)

    def test_comparator_records_first_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.xyz"
            candidate = root / "candidate.xyz"
            reference.write_text(
                "1\nProperties=species:S:1:pos:R:3:velocities:R:3:forces:R:3 energy=0 time_fs=0\n"
                "Li 0 0 0 0 0 0 0 0 0\n"
            )
            candidate.write_text(
                "1\nProperties=species:S:1:pos:R:3:velocities:R:3:forces:R:3 energy=0 time_fs=0\n"
                "Li 0.001 0 0 0 0 0 0 0 0\n"
            )
            args = argparse.Namespace(
                reference=reference,
                candidate=candidate,
                energy_max_ev=1.0e-4,
                position_max_angstrom=1.0e-4,
                velocity_max_angstrom_per_fs=1.0e-5,
                force_max_ev_per_angstrom=5.0e-5,
                force_rms_ev_per_angstrom=1.0e-5,
            )
            result = COMPARATOR.compare(args)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["first_failure"]["frame"], 0)

    def test_propagation_assessment_uses_baseline_noise_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stepwise = root / "stepwise"
            control = root / "control"
            stepwise.mkdir()
            control.mkdir()
            static = root / "static.json"
            static.write_text('{"status":"pass"}\n')
            maxima = {
                "energy_abs_ev": 0.002,
                "position_max_abs_angstrom": 6.0e-6,
                "velocity_max_abs_angstrom_per_fs": 9.0e-7,
                "force_max_abs_ev_per_angstrom": 3.0e-4,
                "force_rms_ev_per_angstrom": 2.5e-5,
                "force_p99_9_abs_ev_per_angstrom": 1.8e-4,
                "force_relative_l2": 1.8e-5,
            }
            control_result = {"maxima": maxima}
            candidate_maxima = dict(maxima)
            candidate_maxima["force_max_abs_ev_per_angstrom"] *= 1.4
            candidate_result = {
                "maxima": candidate_maxima,
                "frame_count_match": True,
                "frames": [{"structural_match": True, "finite": True}],
            }
            for ensemble in ("NVE", "NVT"):
                (control / f"compare_{ensemble}_baseline_repeat.json").write_text(
                    __import__("json").dumps(control_result)
                )
                for backend in ("paired_shared", "paired_stacked"):
                    (stepwise / f"compare_{ensemble}_{backend}.json").write_text(
                        __import__("json").dumps(candidate_result)
                    )
            args = argparse.Namespace(
                static_parity=static,
                stepwise_dir=stepwise,
                control_dir=control,
                position_max_angstrom=1.0e-4,
                velocity_max_angstrom_per_fs=1.0e-5,
                noise_envelope_factor=1.5,
            )
            result = ASSESSOR.assess(args)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(all(case["noise_envelope_gate"] for case in result["cases"]))


if __name__ == "__main__":
    unittest.main()
