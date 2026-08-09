#!/usr/bin/env python3
"""Validate MD_CUDA checkpoint generations and select the newest valid one."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physics_config_sha256(path: Path, workflow_step_index: int) -> str:
    """Hash configuration fields that define continuation physics.

    The target duration and restart/checkpoint controls are deliberately
    excluded so a validated checkpoint can be extended monotonically.
    """
    setting = json.loads(path.read_text())
    for index, step in enumerate(setting.get("steps", [])):
        simulation = step.get("simulation")
        if not isinstance(simulation, dict):
            continue
        if index == workflow_step_index:
            simulation.pop("simulation_time", None)
        simulation.pop("restart", None)
    canonical = json.dumps(setting, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def candidate_sidecars(directory: Path) -> list[Path]:
    return sorted(directory.glob("checkpoint.*.json"), reverse=True)


def validate_sidecar(
    sidecar: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_physics_config_sha256: str | None = None,
    expected_physics_config_path: Path | None = None,
    expected_model_sha256: str | None = None,
    expected_target_step: int | None = None,
    allow_target_extension: bool = False,
) -> dict[str, Any]:
    metadata = json.loads(sidecar.read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version in {sidecar}")
    payload_name = metadata.get("payload_file")
    if not isinstance(payload_name, str) or Path(payload_name).name != payload_name:
        raise ValueError(f"invalid payload_file in {sidecar}")
    payload = sidecar.parent / payload_name
    if not payload.is_file():
        raise FileNotFoundError(payload)
    observed_sha256 = sha256_file(payload)
    if observed_sha256 != metadata.get("payload_sha256"):
        raise ValueError(f"payload SHA-256 mismatch for {payload}")
    if payload.stat().st_size != metadata.get("payload_size"):
        raise ValueError(f"payload size mismatch for {payload}")
    if expected_config_sha256 and metadata.get("config_sha256") != expected_config_sha256:
        raise ValueError(f"config SHA-256 mismatch for {sidecar}")
    if (
        expected_physics_config_sha256
        and metadata.get("physics_config_sha256") != expected_physics_config_sha256
    ):
        raise ValueError(f"physics config SHA-256 mismatch for {sidecar}")
    if expected_physics_config_path:
        workflow_step_index = metadata.get("workflow_step_index")
        if not isinstance(workflow_step_index, int):
            raise ValueError(f"missing workflow_step_index in {sidecar}")
        expected_physics_sha = physics_config_sha256(
            expected_physics_config_path,
            workflow_step_index,
        )
        if metadata.get("physics_config_sha256") != expected_physics_sha:
            raise ValueError(f"physics config SHA-256 mismatch for {sidecar}")
    if expected_model_sha256 and metadata.get("model_sha256") != expected_model_sha256:
        raise ValueError(f"model SHA-256 mismatch for {sidecar}")
    if expected_target_step is not None:
        saved_target_step = metadata.get("target_step")
        current_steps = metadata.get("current_steps")
        if not isinstance(saved_target_step, int) or not isinstance(current_steps, int):
            raise ValueError(f"missing target/current step metadata in {sidecar}")
        if allow_target_extension:
            if (
                expected_target_step < saved_target_step
                or expected_target_step < current_steps
            ):
                raise ValueError(f"target step is not a monotonic extension for {sidecar}")
        elif expected_target_step != saved_target_step:
            raise ValueError(f"target step mismatch for {sidecar}")
    return {
        "sidecar": str(sidecar),
        "payload": str(payload),
        "metadata": metadata,
    }


def discover_latest_valid(
    directory: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_physics_config_sha256: str | None = None,
    expected_physics_config_path: Path | None = None,
    expected_model_sha256: str | None = None,
    expected_target_step: int | None = None,
    allow_target_extension: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    for sidecar in candidate_sidecars(directory):
        try:
            return (
                validate_sidecar(
                    sidecar,
                    expected_config_sha256=expected_config_sha256,
                    expected_physics_config_sha256=expected_physics_config_sha256,
                    expected_physics_config_path=expected_physics_config_path,
                    expected_model_sha256=expected_model_sha256,
                    expected_target_step=expected_target_step,
                    allow_target_extension=allow_target_extension,
                ),
                errors,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append({"sidecar": str(sidecar), "error": str(exc)})
    return None, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--allow-target-extension", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_sha = (
        sha256_file(args.config)
        if args.config and not args.allow_target_extension
        else None
    )
    model_sha = sha256_file(args.model) if args.model else None
    selected, errors = discover_latest_valid(
        args.directory,
        expected_config_sha256=config_sha,
        expected_physics_config_path=(
            args.config if args.config and args.allow_target_extension else None
        ),
        expected_model_sha256=model_sha,
        expected_target_step=args.target_step,
        allow_target_extension=args.allow_target_extension,
    )
    summary = {
        "schema_version": 1,
        "directory": str(args.directory),
        "valid": selected is not None,
        "selected": selected,
        "rejected_generations": errors,
        "fallback_used": bool(selected and errors),
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if selected is None and (args.require or candidate_sidecars(args.directory)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
