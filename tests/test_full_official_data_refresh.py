from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_refresh_dry_run_builds_and_applies_msa_row_filter() -> None:
    proc = subprocess.run(
        [
            "bash",
            "scripts/full_official_data_refresh.sh",
            "--dry-run",
            "--skip-manifest-regen",
            "--skip-fingerprint",
            "--skip-hidden",
            "--download-workers",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = proc.stdout
    filter_path = ".nanofold_private/msa_row_filters/official_heldout_target_homology_filter.json"
    assert "--skip-preprocess" in stdout
    assert "scripts/build_msa_row_filter.py" in stdout
    assert "--source-manifest data/manifests/train.txt" in stdout
    assert "--source-manifest data/manifests/val.txt" in stdout
    assert "--heldout-manifest data/manifests/val.txt" in stdout
    assert f"--output {filter_path}" in stdout
    assert f"--msa-row-filter {filter_path}" in stdout
