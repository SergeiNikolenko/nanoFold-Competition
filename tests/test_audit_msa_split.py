from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from nanofold.a3m import GAP_ID, sequence_to_ids
from nanofold.chain_paths import chain_data_dir, chain_npz_path


def _load_audit_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_msa_split.py"
    spec = importlib.util.spec_from_file_location("audit_msa_split_for_tests", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_feature(features_dir: Path, chain_id: str, rows: list[str]) -> None:
    max_len = max(len(row) for row in rows)
    msa = np.full((len(rows), max_len), GAP_ID, dtype=np.int32)
    for row_index, row in enumerate(rows):
        msa[row_index, : len(row)] = sequence_to_ids(row)
    path = chain_npz_path(features_dir, chain_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        aatype=sequence_to_ids(rows[0]),
        msa=msa,
        deletions=np.zeros_like(msa),
    )


def test_audit_processed_msa_exact_counts_target_leakage(tmp_path: Path) -> None:
    module = _load_audit_module()
    features_dir = tmp_path / "features"
    _write_feature(features_dir, "trn1_A", ["AAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCC"])
    _write_feature(features_dir, "val1_A", ["CCCCCCCCCCCCCCCCCCCC", "DDDDDDDDDDDDDDDDDDDD"])

    summary = module.audit_processed_msa_exact(
        train_ids=["trn1_A"],
        val_ids=["val1_A"],
        features_dir=features_dir,
        include_examples=True,
    )

    assert summary["train"]["rows"] == 2
    assert summary["val"]["rows"] == 2
    assert summary["val_target_sequences_seen_in_train_non_query_msa"] == 1
    assert summary["train_val_non_query_msa_unique_sequence_overlap"] == 0
    assert summary["val_target_sequence_examples"]["val1_A"][0]["chain_id"] == "trn1_A"


def test_msa_depth_distribution_reports_js_divergence() -> None:
    module = _load_audit_module()
    depth_bin_edges = (1, 2, 4)
    train_rows = [
        {"msa_depth": 1, "msa_depth_bin": module._depth_bin(1, depth_bin_edges)},
        {"msa_depth": 2, "msa_depth_bin": module._depth_bin(2, depth_bin_edges)},
    ]
    val_rows = [
        {"msa_depth": 4, "msa_depth_bin": module._depth_bin(4, depth_bin_edges)},
        {"msa_depth": 5, "msa_depth_bin": module._depth_bin(5, depth_bin_edges)},
    ]

    summary = module.summarize_msa_depth_distribution(
        train_rows=train_rows,
        val_rows=val_rows,
        depth_bin_edges=depth_bin_edges,
    )

    assert summary["bins"] == ["<= 1", "2", "3-4", "> 4"]
    assert summary["train_counts"]["<= 1"] == 1
    assert summary["val_counts"]["> 4"] == 1
    assert summary["jensen_shannon_divergence_bits"] > 0


def test_raw_msa_identifier_overlap_canonicalizes_ranges(tmp_path: Path) -> None:
    module = _load_audit_module()
    raw_root = tmp_path / "raw"
    train_dir = chain_data_dir(raw_root / "roda_pdb", "trn1_A")
    val_dir = chain_data_dir(raw_root / "roda_pdb", "val1_A")
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    (train_dir / "uniref90_hits.a3m").write_text(
        ">query\nAAAAAAAAAAAAAAAAAAAA\n"
        ">UniRef90_SHARED/1-20 shared hit\nAAAAAAAAAAAAAAAAAAAA\n"
        ">UniRef90_TRAIN_ONLY/1-20 train hit\nAAAAAAAAAAAAAAAAAAAA\n"
    )
    (val_dir / "uniref90_hits.a3m").write_text(
        ">query\nCCCCCCCCCCCCCCCCCCCC\n"
        ">UniRef90_SHARED/5-24 shared hit different range\nCCCCCCCCCCCCCCCCCCCC\n"
        ">UniRef90_VAL_ONLY/1-20 val hit\nCCCCCCCCCCCCCCCCCCCC\n"
    )

    summary = module.audit_raw_msa_identifier_overlap(
        train_ids=["trn1_A"],
        val_ids=["val1_A"],
        raw_root=raw_root,
        msa_names=("uniref90_hits.a3m",),
        include_examples=True,
    )

    assert summary["train"]["unique_identifiers"] == 2
    assert summary["val"]["unique_identifiers"] == 2
    assert summary["train_val_unique_identifier_overlap"] == 1
    assert summary["overlap_examples"][0]["identifier"] == "UniRef90_SHARED"
