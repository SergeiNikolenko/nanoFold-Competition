"""Write a compact audit/source-lock record for an MSA-row filter.

The row-filter JSON can be very large because it stores every excluded sequence
hash. This helper produces a small, public-safe summary that records the
thresholds, source manifests, MMseqs settings, row-filter hash, and aggregate
processed-feature filtering counts without copying raw sequences, excluded hash
lists, hidden chain IDs, or private paths into the summary by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

PRIVATE_MARKERS = (".nanofold_private", "results_private")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--msa-row-filter", type=Path, required=True)
    ap.add_argument(
        "--processed-filter-audit",
        type=Path,
        action="append",
        default=[],
        help="processed_msa_row_filter_audit.json from each filtering pass. Repeat in order.",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--include-private-paths",
        action="store_true",
        help="Do not redact private paths/hashes. Use only for sealed maintainer-local records.",
    )
    return ap.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def _is_private_path(value: str | None) -> bool:
    if not value:
        return False
    return any(marker in value for marker in PRIVATE_MARKERS)


def _safe_path(value: str | None, *, include_private_paths: bool) -> str | None:
    if value is None:
        return None
    if include_private_paths or not _is_private_path(value):
        return value
    if ".nanofold_private" in value:
        return "<private>"
    return "<private-artifact>"


def _manifest_role(path: str | None) -> str:
    if path is None:
        return "manifest"
    normalized = path.replace("\\", "/").lower()
    if "hidden" in normalized or ".nanofold_private" in normalized:
        return "hidden"
    if normalized.endswith("val.txt"):
        return "public_validation"
    if normalized.endswith("train.txt"):
        return "train"
    return "manifest"


def _safe_manifest(record: Any, *, include_private_paths: bool) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Manifest record must be an object, got {record!r}")
    path = record.get("path")
    if path is not None and not isinstance(path, str):
        raise ValueError(f"Manifest path must be a string, got {path!r}")
    private = _is_private_path(path)
    safe: dict[str, Any] = {
        "role": _manifest_role(path),
        "path": _safe_path(path, include_private_paths=include_private_paths),
        "chain_count": record.get("chain_count"),
    }
    if include_private_paths or not private:
        safe["sha256"] = record.get("sha256")
    else:
        safe["sha256"] = "<private>"
    return safe


def _safe_manifests(records: Any, *, include_private_paths: bool) -> list[dict[str, Any]]:
    if records is None:
        return []
    if not isinstance(records, list):
        raise ValueError("Manifest records must be a list.")
    return [_safe_manifest(record, include_private_paths=include_private_paths) for record in records]


def _normalize_mmseqs_command(command: Any) -> list[str]:
    if not isinstance(command, list):
        return []
    normalized = [str(part) for part in command]
    try:
        search_index = normalized.index("easy-search")
    except ValueError:
        return normalized
    replacements = ["<heldout_targets.fasta>", "<source_msa_rows.fasta>", "<output.m8>", "<tmp_dir>"]
    for offset, replacement in enumerate(replacements, start=1):
        index = search_index + offset
        if index < len(normalized):
            normalized[index] = replacement
    return normalized


def _summarize_row_filter(filter_path: Path, *, include_private_paths: bool) -> dict[str, Any]:
    payload = _load_json(filter_path)
    source = payload.get("source", {})
    heldout = payload.get("heldout", {})
    mmseqs = payload.get("mmseqs", {})
    chain_data_cache = payload.get("chain_data_cache", {})

    if not isinstance(source, dict) or not isinstance(heldout, dict):
        raise ValueError("Row filter is missing source/heldout objects.")
    if not isinstance(mmseqs, dict):
        raise ValueError("Row filter is missing mmseqs object.")
    if not isinstance(chain_data_cache, dict):
        raise ValueError("Row filter has invalid chain_data_cache object.")

    return {
        "path": _safe_path(str(filter_path), include_private_paths=include_private_paths),
        "sha256": _sha256_file(filter_path),
        "filter_type": payload.get("filter_type"),
        "threshold": payload.get("threshold"),
        "excluded_sequence_count": payload.get("excluded_sequence_count"),
        "excluded_sequence_sha256_included": False,
        "mmseqs": {
            "binary": mmseqs.get("binary"),
            "version": mmseqs.get("version"),
            "command": _normalize_mmseqs_command(mmseqs.get("command")),
            "max_seqs": mmseqs.get("max_seqs"),
            "hit_count": mmseqs.get("hit_count"),
        },
        "source": {
            "mode": source.get("mode"),
            "manifests": _safe_manifests(source.get("manifests"), include_private_paths=include_private_paths),
            "raw_root": _safe_path(source.get("raw_root"), include_private_paths=include_private_paths),
            "processed_features_dirs": [
                _safe_path(str(path), include_private_paths=include_private_paths)
                for path in source.get("processed_features_dirs", [])
            ],
            "msa_names": source.get("msa_names"),
            "stats": source.get("stats"),
        },
        "heldout": {
            "manifests": _safe_manifests(heldout.get("manifests"), include_private_paths=include_private_paths),
            "unique_target_sequence_count": heldout.get("unique_target_sequence_count"),
        },
        "chain_data_cache": {
            "path": _safe_path(chain_data_cache.get("path"), include_private_paths=include_private_paths),
            "sha256": chain_data_cache.get("sha256"),
        },
    }


def _summarize_processed_audit(path: Path, *, include_private_paths: bool) -> dict[str, Any]:
    payload = _load_json(path)
    return {
        "path": _safe_path(str(path), include_private_paths=include_private_paths),
        "sha256": _sha256_file(path),
        "operation": payload.get("operation"),
        "out_features_dir": _safe_path(payload.get("out_features_dir"), include_private_paths=include_private_paths),
        "source_processed_features_dirs": [
            _safe_path(str(source_path), include_private_paths=include_private_paths)
            for source_path in payload.get("source_processed_features_dirs", [])
        ],
        "manifests": _safe_manifests(payload.get("manifests"), include_private_paths=include_private_paths),
        "chain_count": payload.get("chain_count"),
        "filtered_chain_count": payload.get("filtered_chain_count"),
        "missing_chain_count": payload.get("missing_chain_count"),
        "excluded_sequence_count": payload.get("excluded_sequence_count"),
        "msa_row_filter_sha256": payload.get("msa_row_filter_sha256"),
        "total_msa_rows_before_filter": payload.get("total_msa_rows_before_filter"),
        "total_msa_rows_after_filter": payload.get("total_msa_rows_after_filter"),
        "total_msa_rows_removed": payload.get("total_msa_rows_removed"),
        "per_chain_included": False,
    }


def _sum_int(records: Sequence[dict[str, Any]], key: str) -> int | None:
    values: list[int] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        values.append(int(value))
    return sum(values) if values else None


def build_source_lock(
    *,
    row_filter_path: Path,
    processed_filter_audit_paths: Sequence[Path],
    include_private_paths: bool = False,
) -> dict[str, Any]:
    filter_summary = _summarize_row_filter(row_filter_path, include_private_paths=include_private_paths)
    audit_summaries = [
        _summarize_processed_audit(path, include_private_paths=include_private_paths)
        for path in processed_filter_audit_paths
    ]
    return {
        "schema_version": 1,
        "artifact_type": "msa_row_filter_source_lock",
        "public_safe": not include_private_paths,
        "safety": {
            "raw_sequences_included": False,
            "excluded_sequence_sha256_included": False,
            "per_chain_processed_audit_included": False,
            "private_paths_redacted": not include_private_paths,
        },
        "msa_row_filter": filter_summary,
        "processed_feature_filter_passes": audit_summaries,
        "aggregate_processed_filtering": {
            "pass_count": len(audit_summaries),
            "cumulative_msa_rows_removed": _sum_int(audit_summaries, "total_msa_rows_removed"),
            "final_total_msa_rows_after_filter": (
                audit_summaries[-1].get("total_msa_rows_after_filter") if audit_summaries else None
            ),
            "final_filtered_chain_count": audit_summaries[-1].get("filtered_chain_count") if audit_summaries else None,
            "final_msa_row_filter_sha256": audit_summaries[-1].get("msa_row_filter_sha256") if audit_summaries else None,
        },
    }


def main() -> None:
    args = parse_args()
    payload = build_source_lock(
        row_filter_path=args.msa_row_filter,
        processed_filter_audit_paths=args.processed_filter_audit,
        include_private_paths=bool(args.include_private_paths),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote MSA row-filter source lock to {args.output}")


if __name__ == "__main__":
    main()
