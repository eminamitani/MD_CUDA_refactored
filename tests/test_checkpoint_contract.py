from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_falls_back_from_corrupt_latest(tmp_path: Path) -> None:
    module = load_script("validate_md_checkpoint.py")
    valid_payload = b"valid"
    valid_name = "checkpoint.step00000000000001.segment0000.g1"
    (tmp_path / f"{valid_name}.bin").write_bytes(valid_payload)
    (tmp_path / f"{valid_name}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "payload_file": f"{valid_name}.bin",
                "payload_sha256": hashlib.sha256(valid_payload).hexdigest(),
                "payload_size": len(valid_payload),
            }
        )
    )
    corrupt_name = "checkpoint.step00000000000002.segment0000.g2"
    (tmp_path / f"{corrupt_name}.bin").write_bytes(b"corrupt")
    (tmp_path / f"{corrupt_name}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "payload_file": f"{corrupt_name}.bin",
                "payload_sha256": "0" * 64,
                "payload_size": 7,
            }
        )
    )
    selected, errors = module.discover_latest_valid(tmp_path)
    assert selected is not None
    assert selected["metadata"]["payload_file"] == f"{valid_name}.bin"
    assert len(errors) == 1


def test_physics_config_hash_ignores_target_and_restart_controls(tmp_path: Path) -> None:
    module = load_script("validate_md_checkpoint.py")
    config = {
        "meta": {"unit": "metal"},
        "steps": [
            {
                "name": "production",
                "simulation": {
                    "dt": 1.0,
                    "simulation_time": 5_000_000.0,
                    "ensemble": {"type": "NVT", "temperature": 800.0},
                    "restart": {
                        "mode": "auto",
                        "max_walltime_seconds": 244800,
                    },
                },
            },
            {
                "name": "followup",
                "simulation": {
                    "dt": 1.0,
                    "simulation_time": 1000.0,
                    "ensemble": {"type": "NVE"},
                },
            },
        ],
    }
    first = tmp_path / "first.json"
    first.write_text(json.dumps(config))
    first_hash = module.physics_config_sha256(first, 0)

    config["steps"][0]["simulation"]["simulation_time"] = 10_000_000.0
    config["steps"][0]["simulation"]["restart"]["max_walltime_seconds"] = 1200
    second = tmp_path / "second.json"
    second.write_text(json.dumps(config))
    assert module.physics_config_sha256(second, 0) == first_hash

    config["steps"][1]["simulation"]["simulation_time"] = 2000.0
    inactive_changed = tmp_path / "inactive_changed.json"
    inactive_changed.write_text(json.dumps(config))
    assert module.physics_config_sha256(inactive_changed, 0) != first_hash
    config["steps"][1]["simulation"]["simulation_time"] = 1000.0

    config["steps"][0]["simulation"]["ensemble"]["temperature"] = 900.0
    third = tmp_path / "third.json"
    third.write_text(json.dumps(config))
    assert module.physics_config_sha256(third, 0) != first_hash


def test_checkpoint_target_may_only_extend_monotonically(tmp_path: Path) -> None:
    module = load_script("validate_md_checkpoint.py")
    payload = b"state"
    stem = "checkpoint.step00000000005000.segment0000.g1"
    (tmp_path / f"{stem}.bin").write_bytes(payload)
    (tmp_path / f"{stem}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "payload_file": f"{stem}.bin",
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_size": len(payload),
                "physics_config_sha256": "a" * 64,
                "current_steps": 5000,
                "target_step": 10000,
            }
        )
    )
    selected, errors = module.discover_latest_valid(
        tmp_path,
        expected_physics_config_sha256="a" * 64,
        expected_target_step=20000,
        allow_target_extension=True,
    )
    assert selected is not None
    assert errors == []

    selected, errors = module.discover_latest_valid(
        tmp_path,
        expected_physics_config_sha256="a" * 64,
        expected_target_step=9000,
        allow_target_extension=True,
    )
    assert selected is None
    assert len(errors) == 1
    assert "not a monotonic extension" in errors[0]["error"]


def write_frame(path: Path, step: int, segment: int, x: float, mode: str = "a") -> None:
    with path.open(mode) as handle:
        handle.write("1\n")
        handle.write(
            f'Properties=species:S:1:pos:R:3 production_step_abs={step} '
            f"segment_id={segment}\n"
        )
        handle.write(f"Li {x} 0 0\n")


def test_segment_reader_deduplicates_matching_boundary(tmp_path: Path) -> None:
    module = load_script("validate_segmented_trajectory.py")
    first = tmp_path / "trajectory.segment0000.extxyz"
    second = tmp_path / "trajectory.segment0001.extxyz"
    write_frame(first, 0, 0, 0.0)
    write_frame(first, 10, 0, 1.0)
    write_frame(second, 10, 1, 1.0)
    write_frame(second, 20, 1, 2.0)
    manifest = tmp_path / "segment_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "segments": [
                    {"segment_id": 0, "trajectory": first.name},
                    {"segment_id": 1, "trajectory": second.name},
                ],
            }
        )
    )
    frames = list(module.iter_manifest_frames(manifest))
    assert [frame.step for frame in frames] == [0, 10, 20]


def test_segment_reader_discards_truncated_tail(tmp_path: Path) -> None:
    module = load_script("validate_segmented_trajectory.py")
    trajectory = tmp_path / "trajectory.segment0000.extxyz"
    write_frame(trajectory, 0, 0, 0.0)
    with trajectory.open("a") as handle:
        handle.write("2\nincomplete\nLi 0 0 0\n")
    frames = list(module.iter_extxyz(trajectory))
    assert len(frames) == 1
