"""Extend an MSA-row filter with residual rows from an MMseqs audit.

``scripts/audit_msa_split.py --run-mmseqs`` writes anonymous source headers in
the form ``trainmsa_<index>`` unless identifier output is explicitly enabled.
This helper reconstructs that same unique source-row order from a processed
feature root, maps residual MMseqs hits back to ungapped sequence hashes, and
writes a new de-identified filter JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.a3m import GAP_ID, MASK_ID, RESTYPES
from nanofold.chain_paths import chain_npz_path
from nanofold.data import read_manifest

MSA_TOKEN_TO_AA = {idx: aa for idx, aa in enumerate(RESTYPES)}
MSA_TOKEN_TO_AA[20] = "X"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-filter", type=Path, required=True)
    ap.add_argument(
        "--source-manifest",
        action="append",
        required=True,
        help="Manifest(s) used as the source side of the audit MMseqs run.",
    )
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--m8", type=Path, required=True, help="MMseqs m8 file from audit_msa_split.py.")
    ap.add_argument("--source-prefix", default="trainmsa_")
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def _manifest_record(path: Path) -> dict[str, Any]:
    chain_ids = read_manifest(path)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "chain_count": len(chain_ids),
    }


def _msa_row_to_ungapped_sequence(row: np.ndarray) -> str:
    chars: list[str] = []
    for value in row.tolist():
        token = int(value)
        if token in (GAP_ID, MASK_ID):
            continue
        chars.append(MSA_TOKEN_TO_AA.get(token, "X"))
    return "".join(chars)


def _source_record_hashes(*, manifest_paths: Sequence[Path], features_dir: Path) -> list[str]:
    source_hashes: list[str] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        for chain_id in read_manifest(manifest_path):
            feature_path = chain_npz_path(features_dir, chain_id)
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing processed feature NPZ for {chain_id}: {feature_path}")
            with np.load(feature_path) as data:
                msa = np.asarray(data["msa"])
            for row in msa[1:]:
                sequence = _msa_row_to_ungapped_sequence(row)
                if len(sequence) < 20:
                    continue
                digest = _sequence_sha256(sequence)
                if digest in seen:
                    continue
                seen.add(digest)
                source_hashes.append(digest)
    return source_hashes


def _parse_residual_source_indices(m8_path: Path, *, source_prefix: str) -> tuple[set[int], int]:
    indices: set[int] = set()
    hit_count = 0
    for line in m8_path.read_text().splitlines():
        if not line.strip():
            continue
        hit_count += 1
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"Malformed MMseqs row in {m8_path}: {line!r}")
        source_header = parts[1].split("|", 1)[0]
        if not source_header.startswith(source_prefix):
            raise ValueError(
                f"MMseqs target header does not start with {source_prefix!r}: {source_header!r}"
            )
        raw_index = source_header.removeprefix(source_prefix)
        try:
            indices.add(int(raw_index))
        except ValueError as exc:
            raise ValueError(f"Bad MMseqs source index in {source_header!r}") from exc
    return indices, hit_count


def extend_filter_from_audit(
    *,
    base_filter_path: Path,
    source_manifest_paths: Sequence[Path],
    features_dir: Path,
    m8_path: Path,
    source_prefix: str,
    output_path: Path,
) -> dict[str, Any]:
    payload = json.loads(base_filter_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Base filter must be a JSON object: {base_filter_path}")
    raw_excluded = payload.get("excluded_sequence_sha256")
    if not isinstance(raw_excluded, list):
        raise ValueError(f"Base filter missing `excluded_sequence_sha256`: {base_filter_path}")

    base_excluded = {str(value).lower() for value in raw_excluded}
    bad_hashes = [value for value in base_excluded if len(value) != 64]
    if bad_hashes:
        raise ValueError(f"Base filter contains malformed SHA256 values: {bad_hashes[:3]}")

    source_hashes = _source_record_hashes(manifest_paths=source_manifest_paths, features_dir=features_dir)
    residual_indices, hit_count = _parse_residual_source_indices(m8_path, source_prefix=source_prefix)
    out_of_range = [index for index in residual_indices if index < 0 or index >= len(source_hashes)]
    if out_of_range:
        raise ValueError(
            f"Residual MMseqs source indices exceed reconstructed source records: {out_of_range[:8]}"
        )

    residual_hashes = {source_hashes[index] for index in residual_indices}
    merged = base_excluded | residual_hashes
    supplemental_record = {
        "type": "mmseqs_audit_residual_closure",
        "base_filter_path": str(base_filter_path),
        "base_filter_sha256": _sha256_file(base_filter_path),
        "features_dir": str(features_dir),
        "source_manifests": [_manifest_record(path) for path in source_manifest_paths],
        "m8_path": str(m8_path),
        "m8_sha256": _sha256_file(m8_path),
        "source_prefix": source_prefix,
        "source_record_count": len(source_hashes),
        "m8_hit_count": hit_count,
        "residual_source_record_count": len(residual_indices),
        "new_excluded_sequence_count": len(residual_hashes - base_excluded),
        "already_excluded_sequence_count": len(residual_hashes & base_excluded),
    }

    payload = dict(payload)
    payload["excluded_sequence_sha256"] = sorted(merged)
    payload["excluded_sequence_count"] = len(merged)
    supplemental = payload.get("supplemental_filters")
    if not isinstance(supplemental, list):
        supplemental = []
    supplemental.append(supplemental_record)
    payload["supplemental_filters"] = supplemental

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    args = parse_args()
    payload = extend_filter_from_audit(
        base_filter_path=args.base_filter,
        source_manifest_paths=[Path(path) for path in args.source_manifest],
        features_dir=args.features_dir,
        m8_path=args.m8,
        source_prefix=str(args.source_prefix),
        output_path=args.output,
    )
    latest = payload["supplemental_filters"][-1]
    print(
        "Wrote extended MSA row filter: "
        f"+{latest['new_excluded_sequence_count']} new hashes, "
        f"{payload['excluded_sequence_count']} total -> {args.output}"
    )


if __name__ == "__main__":
    main()
