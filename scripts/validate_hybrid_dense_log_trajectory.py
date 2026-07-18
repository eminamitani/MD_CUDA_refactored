#!/usr/bin/env python3
"""Validate the hybrid dense/log/burst plus uniform extxyz schedule."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


STEP_RE = re.compile(r"(?:^|\s)step_rel=(-?\d+)(?:\s|$)")
SAMPLE_RE = re.compile(r"(?:^|\s)sample_type=([^\s]+)")
KNOWN_SAMPLE_TYPES = {"initial", "dense", "anchor", "burst", "linear"}


def read_comments(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    comments: list[str] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        natoms = int(lines[cursor])
        if cursor + natoms + 1 >= len(lines):
            raise ValueError(f"{path}: truncated extxyz frame at line {cursor + 1}")
        comments.append(lines[cursor + 1])
        cursor += natoms + 2
    if not comments:
        raise ValueError(f"{path}: no frames")
    return comments


def validate_hybrid_schedule(
    path: Path,
    *,
    linear_interval: int,
    total_steps: int,
    dense_until: int | None = None,
) -> dict[str, object]:
    if linear_interval <= 0:
        raise ValueError("linear_interval must be positive")
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")

    samples: list[tuple[int, str]] = []
    for comment in read_comments(path):
        step_match = STEP_RE.search(comment)
        sample_match = SAMPLE_RE.search(comment)
        if step_match is None or sample_match is None:
            raise ValueError(f"{path}: missing step_rel or sample_type metadata")
        step = int(step_match.group(1))
        sample_type = sample_match.group(1)
        if sample_type not in KNOWN_SAMPLE_TYPES:
            raise ValueError(f"{path}: unknown sample_type={sample_type}")
        if sample_type == "linear" and step % linear_interval:
            raise ValueError(f"{path}: off-grid linear sample at step {step}")
        samples.append((step, sample_type))

    steps = [step for step, _ in samples]
    if steps != sorted(steps):
        raise ValueError(f"{path}: step_rel is not monotonic")
    if len(steps) != len(set(steps)):
        raise ValueError(f"{path}: duplicate step_rel values")

    observed = set(steps)
    expected_grid = set(range(0, total_steps + 1, linear_interval))
    missing_grid = sorted(expected_grid - observed)
    if missing_grid:
        raise ValueError(f"{path}: missing uniform-grid steps {missing_grid[:10]}")

    if dense_until is not None:
        expected_dense = set(range(0, min(dense_until, total_steps + 1)))
        missing_dense = sorted(expected_dense - observed)
        if missing_dense:
            raise ValueError(f"{path}: missing dense steps {missing_dense[:10]}")

    counts = {
        sample_type: sum(observed_type == sample_type for _, observed_type in samples)
        for sample_type in sorted(KNOWN_SAMPLE_TYPES)
    }
    if counts["linear"] == 0:
        raise ValueError(f"{path}: no linear fallback samples")
    if counts["anchor"] == 0 or counts["burst"] == 0:
        raise ValueError(f"{path}: no anchor/burst samples")

    return {
        "frames": len(samples),
        "step_min": min(steps),
        "step_max": max(steps),
        "uniform_grid_frames": len(expected_grid),
        "sample_type_counts": counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--linear-interval", required=True, type=int)
    parser.add_argument("--total-steps", required=True, type=int)
    parser.add_argument("--dense-until", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        validate_hybrid_schedule(
            args.trajectory,
            linear_interval=args.linear_interval,
            total_steps=args.total_steps,
            dense_until=args.dense_until,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
