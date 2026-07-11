from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_trajectory_output_modes.py"
SPEC = importlib.util.spec_from_file_location("validate_trajectory_output_modes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def frame(properties: str, rows: list[str], time_fs: float, pbc: str = "T T T") -> str:
    return (
        f"{len(rows)}\n"
        f'Lattice="10 0 0 0 10 0 0 0 10" Properties={properties} '
        f'pbc="{pbc}" trajectory_mode=test time_fs={time_fs}\n'
        + "\n".join(rows)
        + "\n"
    )


class TrajectoryOutputModeValidatorTests(unittest.TestCase):
    def test_validates_uniform_vdos_and_velocity_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vdos.xyz"
            path.write_text(
                frame(
                    "species:S:1:pos:R:3:velocities:R:3",
                    ["Li 0 0 0 0.01 0.02 0.03", "O 1 1 1 -0.01 0.01 0.02"],
                    0.0,
                )
                + frame(
                    "species:S:1:pos:R:3:velocities:R:3",
                    ["Li 0 0 0 0.02 0.02 0.03", "O 1 1 1 -0.01 0.02 0.02"],
                    0.5,
                )
                + frame(
                    "species:S:1:pos:R:3:velocities:R:3",
                    ["Li 0 0 0 0.03 0.02 0.03", "O 1 1 1 -0.01 0.03 0.02"],
                    1.0,
                ),
                encoding="utf-8",
            )

            summary = validator.validate_file(path, ["pos", "velocities"], True)
            diagnostics = validator.velocity_diagnostics(path)

        self.assertEqual(summary["frames"], 3)
        self.assertTrue(diagnostics["fft_power_k1"] >= 0.0)

    def test_rejects_nonuniform_vdos_time_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vdos.xyz"
            content = ""
            for time_fs in (0.0, 0.5, 1.5):
                content += frame(
                    "species:S:1:pos:R:3:velocities:R:3",
                    ["Li 0 0 0 0.01 0.02 0.03"],
                    time_fs,
                )
            path.write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-uniform"):
                validator.validate_file(path, ["pos", "velocities"], True)


if __name__ == "__main__":
    unittest.main()
