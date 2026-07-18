from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_hybrid_dense_log_trajectory.py"
SPEC = importlib.util.spec_from_file_location("validate_hybrid_dense_log_trajectory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def frame(step: int, sample_type: str) -> str:
    return (
        "1\n"
        f"Properties=species:S:1:pos:R:3 step_rel={step} time_fs={step * 0.5} "
        f"sample_type={sample_type}\n"
        "Li 0 0 0\n"
    )


class HybridDenseLogTrajectoryTests(unittest.TestCase):
    def test_accepts_dense_anchor_burst_and_uniform_union(self) -> None:
        samples = [
            (0, "initial"),
            (1, "dense"),
            (2, "dense"),
            (3, "dense"),
            (4, "dense"),
            (5, "anchor"),
            (6, "burst"),
            (7, "burst"),
            (8, "burst"),
            (10, "linear"),
            (11, "anchor"),
            (12, "burst"),
            (13, "burst"),
            (14, "burst"),
            (15, "linear"),
            (20, "linear"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hybrid.extxyz"
            path.write_text("".join(frame(*sample) for sample in samples), encoding="utf-8")
            summary = validator.validate_hybrid_schedule(
                path,
                linear_interval=5,
                total_steps=20,
                dense_until=5,
            )
        self.assertEqual(summary["frames"], len(samples))
        self.assertEqual(summary["uniform_grid_frames"], 5)
        self.assertEqual(summary["sample_type_counts"]["linear"], 3)

    def test_accepts_schedule_collision_without_duplicate_frame(self) -> None:
        samples = [
            (0, "initial"),
            (1, "dense"),
            (2, "dense"),
            (3, "dense"),
            (4, "dense"),
            (5, "anchor"),
            (6, "burst"),
            (7, "burst"),
            (8, "burst"),
            (10, "anchor"),
            (11, "burst"),
            (12, "burst"),
            (13, "burst"),
            (15, "linear"),
            (20, "linear"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "collision.extxyz"
            path.write_text("".join(frame(*sample) for sample in samples), encoding="utf-8")
            summary = validator.validate_hybrid_schedule(
                path,
                linear_interval=5,
                total_steps=20,
                dense_until=5,
            )
        self.assertEqual(summary["uniform_grid_frames"], 5)

    def test_rejects_missing_uniform_grid_step(self) -> None:
        samples = [
            (0, "initial"),
            (1, "dense"),
            (2, "dense"),
            (3, "dense"),
            (4, "dense"),
            (5, "anchor"),
            (6, "burst"),
            (7, "burst"),
            (8, "burst"),
            (15, "linear"),
            (20, "linear"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.extxyz"
            path.write_text("".join(frame(*sample) for sample in samples), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing uniform-grid"):
                validator.validate_hybrid_schedule(
                    path,
                    linear_interval=5,
                    total_steps=20,
                    dense_until=5,
                )

    def test_rejects_duplicate_step(self) -> None:
        samples = [
            (0, "initial"),
            (1, "dense"),
            (2, "dense"),
            (3, "dense"),
            (4, "dense"),
            (5, "anchor"),
            (5, "linear"),
            (6, "burst"),
            (7, "burst"),
            (8, "burst"),
            (10, "linear"),
            (15, "linear"),
            (20, "linear"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.extxyz"
            path.write_text("".join(frame(*sample) for sample in samples), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validator.validate_hybrid_schedule(
                    path,
                    linear_interval=5,
                    total_steps=20,
                    dense_until=5,
                )


if __name__ == "__main__":
    unittest.main()
