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
