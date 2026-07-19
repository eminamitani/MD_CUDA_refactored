#!/usr/bin/env python3
"""Validate segment manifests and stream extxyz frames without physical concatenation."""

from __future__ import annotations

import argparse
import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Frame:
    path: Path
    index: int
    symbols: tuple[str, ...]
    positions: np.ndarray
    metadata: dict[str, str]

    @property
    def step(self) -> int:
        return int(self.metadata["production_step_abs"])


def parse_comment(comment: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for token in shlex.split(comment):
        if "=" in token:
            key, value = token.split("=", 1)
            metadata[key] = value
    return metadata


def iter_extxyz(path: Path, *, discard_truncated_tail: bool = True) -> Iterator[Frame]:
    with path.open() as handle:
        frame_index = 0
        while True:
            first = handle.readline()
            if not first:
                return
            try:
                atom_count = int(first.strip())
            except ValueError as exc:
                if discard_truncated_tail:
                    return
                raise ValueError(f"{path}: invalid atom count at frame {frame_index}") from exc
            comment = handle.readline()
            if not comment:
                if discard_truncated_tail:
                    return
                raise ValueError(f"{path}: truncated comment at frame {frame_index}")
            rows = [handle.readline() for _ in range(atom_count)]
            if any(not row for row in rows):
                if discard_truncated_tail:
                    return
                raise ValueError(f"{path}: truncated atom rows at frame {frame_index}")
            symbols: list[str] = []
            positions = np.empty((atom_count, 3), dtype=np.float64)
            for atom_index, row in enumerate(rows):
                fields = row.split()
                if len(fields) < 4:
                    if discard_truncated_tail:
                        return
                    raise ValueError(f"{path}: malformed atom row at frame {frame_index}")
                symbols.append(fields[0])
                positions[atom_index] = [float(value) for value in fields[1:4]]
            metadata = parse_comment(comment)
            if "production_step_abs" not in metadata or "segment_id" not in metadata:
                raise ValueError(f"{path}: missing restart metadata at frame {frame_index}")
            yield Frame(path, frame_index, tuple(symbols), positions, metadata)
            frame_index += 1


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("segments"), list):
        raise ValueError("invalid segment manifest")
    return manifest


def iter_manifest_frames(
    manifest_path: Path,
    *,
    coordinate_tolerance: float = 1.0e-6,
) -> Iterator[Frame]:
    manifest = load_manifest(manifest_path)
    previous: Frame | None = None
    for segment in sorted(manifest["segments"], key=lambda item: item["segment_id"]):
        trajectory = Path(segment["trajectory"])
        if not trajectory.is_absolute():
            trajectory = manifest_path.parent / trajectory
        expected_segment = int(segment["segment_id"])
        for frame in iter_extxyz(trajectory):
            if int(frame.metadata["segment_id"]) != expected_segment:
                raise ValueError(f"{trajectory}: segment_id mismatch")
            if previous is not None:
                if frame.step < previous.step:
                    raise ValueError(f"non-monotonic step {frame.step} after {previous.step}")
                if frame.step == previous.step:
                    if frame.symbols != previous.symbols:
                        raise ValueError(f"boundary species mismatch at step {frame.step}")
                    difference = float(np.max(np.abs(frame.positions - previous.positions)))
                    if not math.isfinite(difference) or difference > coordinate_tolerance:
                        raise ValueError(
                            f"boundary coordinate mismatch at step {frame.step}: {difference}"
                        )
                    continue
            previous = frame
            yield frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--coordinate-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--expected-linear-interval", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frames = list(
        iter_manifest_frames(
            args.manifest,
            coordinate_tolerance=args.coordinate_tolerance,
        )
    )
    steps = [frame.step for frame in frames]
    linear_steps = []
    if args.expected_linear_interval:
        linear_steps = [
            step for step in steps if step % args.expected_linear_interval == 0
        ]
        if linear_steps:
            expected = list(
                range(
                    linear_steps[0],
                    linear_steps[-1] + args.expected_linear_interval,
                    args.expected_linear_interval,
                )
            )
            if linear_steps != expected:
                raise ValueError("uniform linear samples contain a gap")
    summary = {
        "schema_version": 1,
        "valid": True,
        "frame_count": len(frames),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "linear_frame_count": len(linear_steps),
        "segments": sorted({int(frame.metadata["segment_id"]) for frame in frames}),
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
