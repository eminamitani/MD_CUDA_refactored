#!/usr/bin/env python3
"""Parse one variable-shape PaiNN timing run and measurement thermodynamics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


ELAPSED_RE = re.compile(r"かかった時間：([0-9.eE+-]+)s")
TELEMETRY_RE = re.compile(r"NNP edge telemetry: samples=(\d+), min=(\d+), max=(\d+), mean=([0-9.eE+-]+)")
MEASUREMENT_MARKER = "シミュレーション: measurement_9ps"


def gpu_metrics(path: Path) -> dict[str, float | int | str | None]:
    rows: list[list[str]] = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = [[field.strip() for field in row] for row in csv.reader(handle) if len(row) >= 7]
    if not rows:
        return {"samples": 0}

    def values(index: int) -> list[float]:
        result: list[float] = []
        for row in rows:
            try:
                result.append(float(row[index]))
            except ValueError:
                pass
        return result

    clock, util, power, temperature = values(3), values(4), values(5), values(6)
    return {
        "samples": len(rows),
        "gpu_name": rows[0][2],
        "clock_sm_mhz_mean": statistics.fmean(clock) if clock else None,
        "utilization_percent_mean": statistics.fmean(util) if util else None,
        "power_w_mean": statistics.fmean(power) if power else None,
        "temperature_c_mean": statistics.fmean(temperature) if temperature else None,
    }


def thermo_rows(text: str) -> list[list[float]]:
    measurement_text = text.split(MEASUREMENT_MARKER, maxsplit=1)[-1]
    rows: list[list[float]] = []
    for line in measurement_text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            rows.append([float(field) for field in fields])
        except ValueError:
            pass
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--gpu-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("legacy", "md_baseline", "md_sync", "md_csr"))
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--order-index", required=True, type=int)
    parser.add_argument("--measure-steps", type=int, default=18000)
    parser.add_argument("--measure-fs", type=float, default=9000.0)
    parser.add_argument("--returncode", required=True, type=int)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    elapsed = [float(value) for value in ELAPSED_RE.findall(text)]
    measurement_seconds = elapsed[1] if len(elapsed) >= 2 else None
    rows = thermo_rows(text)
    finite = bool(rows) and all(math.isfinite(value) for row in rows for value in row)
    telemetry = TELEMETRY_RE.search(text)

    def column(index: int) -> list[float]:
        return [row[index] for row in rows]

    summary = {
        "backend": args.backend,
        "repeat": args.repeat,
        "order_index": args.order_index,
        "returncode": args.returncode,
        "warmup_seconds": elapsed[0] if elapsed else None,
        "measurement_seconds": measurement_seconds,
        "measure_steps": args.measure_steps,
        "measure_fs": args.measure_fs,
        "steps_per_second": args.measure_steps / measurement_seconds if measurement_seconds else None,
        "ns_per_day": (args.measure_fs / 1.0e6) * 86400.0 / measurement_seconds if measurement_seconds else None,
        "finite_thermodynamics": finite,
        "thermo_samples": len(rows),
        "mean_potential_energy_ev": statistics.fmean(column(2)) if rows else None,
        "mean_total_energy_ev": statistics.fmean(column(3)) if rows else None,
        "mean_temperature_k": statistics.fmean(column(4)) if rows else None,
        "std_total_energy_ev": statistics.pstdev(column(3)) if len(rows) > 1 else 0.0 if rows else None,
        "std_temperature_k": statistics.pstdev(column(4)) if len(rows) > 1 else 0.0 if rows else None,
        "edge_samples": int(telemetry.group(1)) if telemetry else None,
        "edge_min": int(telemetry.group(2)) if telemetry else None,
        "edge_max": int(telemetry.group(3)) if telemetry else None,
        "edge_mean": float(telemetry.group(4)) if telemetry else None,
        "gpu": gpu_metrics(args.gpu_csv),
    }
    summary["valid"] = bool(
        args.returncode == 0
        and measurement_seconds is not None
        and measurement_seconds > 0
        and finite
    )
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
