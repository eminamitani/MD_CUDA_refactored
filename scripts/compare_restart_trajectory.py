#!/usr/bin/env python3
"""Compare an uninterrupted extxyz trajectory with segmented restart output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def read_frames(path: Path) -> list[dict]:
    frames: list[dict] = []
    with path.open() as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            atom_count = int(first)
            comment = handle.readline().strip()
            match = re.search(r"Properties=([^ ]+)", comment)
            if not match:
                raise ValueError(f"{path}: missing Properties")
            schema = match.group(1).split(":")
            columns: list[tuple[str, int]] = []
            index = 0
            while index < len(schema):
                columns.append((schema[index], int(schema[index + 2])))
                index += 3
            symbols: list[str] = []
            positions = np.empty((atom_count, 3))
            velocities = np.empty((atom_count, 3))
            has_velocity = False
            for atom_index in range(atom_count):
                values = handle.readline().split()
                offset = 0
                for name, width in columns:
                    block = values[offset : offset + width]
                    offset += width
                    if name == "species":
                        symbols.append(block[0])
                    elif name == "pos":
                        positions[atom_index] = [float(value) for value in block]
                    elif name == "velocities":
                        velocities[atom_index] = [float(value) for value in block]
                        has_velocity = True
            frames.append(
                {
                    "symbols": tuple(symbols),
                    "positions": positions,
                    "velocities": velocities if has_velocity else None,
                }
            )
    return frames


def read_segmented(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    frames: list[dict] = []
    for segment in sorted(manifest["segments"], key=lambda item: item["segment_id"]):
        path = Path(segment["trajectory"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        frames.extend(read_frames(path))
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--segment-manifest", type=Path, required=True)
    parser.add_argument("--position-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--velocity-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    continuous = read_frames(args.continuous)
    restarted = read_segmented(args.segment_manifest)
    if len(continuous) != len(restarted):
        raise ValueError(f"frame count mismatch: {len(continuous)} != {len(restarted)}")
    max_position = 0.0
    max_velocity = 0.0
    for index, (left, right) in enumerate(zip(continuous, restarted)):
        if left["symbols"] != right["symbols"]:
            raise ValueError(f"species mismatch at frame {index}")
        max_position = max(
            max_position,
            float(np.max(np.abs(left["positions"] - right["positions"]))),
        )
        if left["velocities"] is not None or right["velocities"] is not None:
            if left["velocities"] is None or right["velocities"] is None:
                raise ValueError(f"velocity schema mismatch at frame {index}")
            max_velocity = max(
                max_velocity,
                float(np.max(np.abs(left["velocities"] - right["velocities"]))),
            )
    summary = {
        "schema_version": 1,
        "frame_count": len(continuous),
        "max_position_difference_angstrom": max_position,
        "max_velocity_difference_angstrom_per_fs": max_velocity,
        "position_tolerance": args.position_tolerance,
        "velocity_tolerance": args.velocity_tolerance,
        "passed": max_position <= args.position_tolerance
        and max_velocity <= args.velocity_tolerance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
