#!/usr/bin/env python3
"""Compute multi-origin species MSD directly from a segment manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from validate_segmented_trajectory import iter_manifest_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dt-fs", type=float, required=True)
    parser.add_argument("--linear-interval", type=int, required=True)
    parser.add_argument("--max-lag-frames", type=int)
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selected = [
        frame
        for frame in iter_manifest_frames(args.manifest)
        if frame.step % args.linear_interval == 0
    ]
    if len(selected) < 3:
        raise ValueError("at least three uniform frames are required")
    symbols = np.asarray(selected[0].symbols)
    if any(tuple(symbols) != frame.symbols for frame in selected[1:]):
        raise ValueError("atom ordering changed between segments")
    positions = np.stack([frame.positions for frame in selected])
    max_lag = args.max_lag_frames or (len(selected) - 1)
    max_lag = min(max_lag, len(selected) - 1)
    if args.block_count < 1:
        raise ValueError("block-count must be positive")
    species = sorted(set(symbols.tolist()))
    rows: list[dict[str, float | int | str]] = []
    for lag in range(1, max_lag + 1):
        displacement = positions[lag:] - positions[:-lag]
        squared = np.sum(displacement * displacement, axis=2)
        for element in species:
            values = squared[:, symbols == element]
            rows.append(
                {
                    "block_id": -1,
                    "species": element,
                    "lag_frames": lag,
                    "lag_steps": lag * args.linear_interval,
                    "lag_fs": lag * args.linear_interval * args.dt_fs,
                    "msd_angstrom2": float(np.mean(values)),
                    "origin_count": int(values.shape[0]),
                    "atom_count": int(values.shape[1]),
                }
            )
            for block_id, block in enumerate(np.array_split(values, args.block_count, axis=0)):
                if block.shape[0] == 0:
                    continue
                rows.append(
                    {
                        "block_id": block_id,
                        "species": element,
                        "lag_frames": lag,
                        "lag_steps": lag * args.linear_interval,
                        "lag_fs": lag * args.linear_interval * args.dt_fs,
                        "msd_angstrom2": float(np.mean(block)),
                        "origin_count": int(block.shape[0]),
                        "atom_count": int(block.shape[1]),
                    }
                )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "uniform_frame_count": len(selected),
        "first_step": selected[0].step,
        "last_step": selected[-1].step,
        "linear_interval": args.linear_interval,
        "dt_fs": args.dt_fs,
        "block_count": args.block_count,
        "species": species,
        "output_csv": str(args.output_csv),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
