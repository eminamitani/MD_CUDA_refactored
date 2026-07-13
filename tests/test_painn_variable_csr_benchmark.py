from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from parse_painn_variable_csr_benchmark_run import thermo_rows


class PainnVariableCSRBenchmarkTests(unittest.TestCase):
    def test_generator_selects_variable_csr_and_100_step_thermo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "config.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_painn_variable_csr_benchmark_config.py"),
                    "--output",
                    str(output),
                    "--name",
                    "csr-test",
                    "--input-xyz",
                    "/tmp/input.xyz",
                    "--model",
                    "/tmp/model.pt",
                    "--potential-type",
                    "NNP_csr",
                    "--max-edges",
                    "200000",
                ],
                check=True,
            )
            config = json.loads(output.read_text())
            potential = config["common_settings"]["interactions"]["potentials"]
            self.assertEqual(potential["type"], "NNP_csr")
            self.assertEqual(config["steps"][1]["observer"]["interval"], 100)
            self.assertNotIn("trajectory", config["steps"][1]["observer"])

    def test_thermo_parser_uses_measurement_section_only(self) -> None:
        text = """
シミュレーション: warmup_1psを実行します。
0.0, 1.0, -2.0, -1.0, 1000.0
シミュレーション: measurement_9psを実行します。
0.0, 2.0, -3.0, -1.0, 1500.0
50.0, 2.1, -3.1, -1.0, 1510.0
"""
        rows = thermo_rows(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][4], 1500.0)
        self.assertEqual(rows[1][4], 1510.0)


if __name__ == "__main__":
    unittest.main()
