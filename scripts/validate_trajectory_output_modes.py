#!/usr/bin/env python3
"""Validate MD_CUDA extxyz field modes after a GPU smoke run."""

from __future__ import annotations

import argparse
import cmath
import math
import re
from pathlib import Path


PROPERTY_RE = re.compile(r"(?:^|\s)Properties=([^\s]+)")
TIME_RE = re.compile(r"(?:^|\s)time_fs=([^\s]+)")
MASS_AMU = {
    "Li": 6.94,
    "O": 15.999,
    "Na": 22.98976928,
    "Si": 28.085,
    "K": 39.0983,
}
CONVERSION_FACTOR = 0.964855e-2
KB_EV_K = 8.617333262145e-5


def read_extxyz(path: Path) -> list[tuple[str, list[list[str]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[tuple[str, list[list[str]]]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        natoms = int(lines[cursor])
        if cursor + natoms + 1 >= len(lines):
            raise ValueError(f"truncated extxyz frame in {path} at line {cursor + 1}")
        comment = lines[cursor + 1]
        atoms = [line.split() for line in lines[cursor + 2 : cursor + 2 + natoms]]
        if len(atoms) != natoms:
            raise ValueError(f"atom count mismatch in {path}")
        frames.append((comment, atoms))
        cursor += natoms + 2
    if not frames:
        raise ValueError(f"no frames found in {path}")
    return frames


def properties(comment: str) -> list[tuple[str, str, int]]:
    match = PROPERTY_RE.search(comment)
    if match is None:
        raise ValueError("missing Properties entry")
    tokens = match.group(1).split(":")
    if len(tokens) % 3:
        raise ValueError(f"malformed Properties entry: {match.group(1)}")
    return [
        (tokens[index], tokens[index + 1], int(tokens[index + 2]))
        for index in range(0, len(tokens), 3)
    ]


def field_offset(schema: list[tuple[str, str, int]], name: str) -> tuple[int, int]:
    offset = 0
    for field, _, width in schema:
        if field == name:
            return offset, width
        offset += width
    raise ValueError(f"field {name!r} not found")


def validate_file(path: Path, expected_fields: list[str], require_uniform_time: bool) -> dict[str, float | int]:
    frames = read_extxyz(path)
    first_schema = properties(frames[0][0])
    observed_fields = [name for name, _, _ in first_schema]
    if observed_fields != ["species", *expected_fields]:
        raise ValueError(f"{path}: fields {observed_fields}, expected {['species', *expected_fields]}")

    times: list[float] = []
    for comment, atoms in frames:
        if properties(comment) != first_schema:
            raise ValueError(f"{path}: schema changes between frames")
        match = TIME_RE.search(comment)
        if match is None:
            raise ValueError(f"{path}: missing time_fs metadata")
        times.append(float(match.group(1)))
        expected_columns = sum(width for _, _, width in first_schema)
        for row in atoms:
            if len(row) != expected_columns:
                raise ValueError(f"{path}: expected {expected_columns} columns, got {len(row)}")
            for value in row[1:]:
                if not math.isfinite(float(value)):
                    raise ValueError(f"{path}: non-finite atom value {value}")

    if require_uniform_time and len(times) >= 3:
        intervals = [right - left for left, right in zip(times, times[1:])]
        if max(intervals) - min(intervals) > 1e-10 * max(1.0, abs(intervals[0])):
            raise ValueError(f"{path}: non-uniform frame intervals {intervals}")
    return {"frames": len(frames), "time_min_fs": min(times), "time_max_fs": max(times)}


def reconstructed_temperature(path: Path, degrees_of_freedom: int | None) -> float:
    comment, atoms = read_extxyz(path)[0]
    schema = properties(comment)
    velocity_offset, velocity_width = field_offset(schema, "velocities")
    if velocity_width != 3:
        raise ValueError("velocities must have width 3")
    kinetic = 0.0
    for row in atoms:
        species = row[0]
        if species not in MASS_AMU:
            raise ValueError(f"no mass configured for species {species}")
        start = velocity_offset
        vx, vy, vz = (float(value) for value in row[start : start + 3])
        kinetic += 0.5 * MASS_AMU[species] * (vx * vx + vy * vy + vz * vz)
    kinetic /= CONVERSION_FACTOR
    dof = degrees_of_freedom or 3 * len(atoms)
    return 2.0 * kinetic / (dof * KB_EV_K)


def velocity_diagnostics(path: Path) -> dict[str, float]:
    frames = read_extxyz(path)
    schema = properties(frames[0][0])
    velocity_offset, velocity_width = field_offset(schema, "velocities")
    if velocity_width != 3:
        raise ValueError("velocities must have width 3")
    velocity_frames: list[list[float]] = []
    for _, atoms in frames:
        flattened: list[float] = []
        for row in atoms:
            flattened.extend(float(value) for value in row[velocity_offset : velocity_offset + 3])
        velocity_frames.append(flattened)

    reference = velocity_frames[0]
    vacf_lag1 = sum(a * b for a, b in zip(reference, velocity_frames[min(1, len(frames) - 1)])) / len(reference)
    frequency_index = 1 if len(frames) > 1 else 0
    power = 0.0
    for component in range(len(reference)):
        amplitude = sum(
            frame[component] * cmath.exp(-2j * math.pi * frequency_index * index / len(frames))
            for index, frame in enumerate(velocity_frames)
        )
        power += abs(amplitude) ** 2
    power /= max(1, len(reference) * len(frames))
    if not math.isfinite(vacf_lag1) or not math.isfinite(power):
        raise ValueError(f"{path}: non-finite VACF/FFT diagnostic")
    return {"vacf_lag1": vacf_lag1, "fft_power_k1": power}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vdos", required=True, type=Path)
    parser.add_argument("--msd", required=True, type=Path)
    parser.add_argument("--transport-base", required=True, type=Path)
    parser.add_argument("--expected-initial-temperature-k", type=float)
    parser.add_argument("--temperature-dof", type=int)
    parser.add_argument("--temperature-tolerance-fraction", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = {
        "vdos": validate_file(args.vdos, ["pos", "velocities"], True),
        "msd": validate_file(args.msd, ["pos"], True),
        "transport_base": validate_file(
            args.transport_base,
            ["pos", "velocities", "forces"],
            True,
        ),
    }
    summaries["vdos"].update(velocity_diagnostics(args.vdos))
    temperature = reconstructed_temperature(args.vdos, args.temperature_dof)
    if args.expected_initial_temperature_k is not None:
        relative_error = abs(temperature - args.expected_initial_temperature_k) / args.expected_initial_temperature_k
        if relative_error > args.temperature_tolerance_fraction:
            raise ValueError(
                f"reconstructed initial temperature {temperature:.8g} K differs from "
                f"expected {args.expected_initial_temperature_k:.8g} K by {relative_error:.3%}"
            )
    print({"summaries": summaries, "reconstructed_initial_temperature_k": temperature})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
