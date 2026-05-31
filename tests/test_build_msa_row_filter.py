from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from nanofold.a3m import GAP_ID, sequence_to_ids
from nanofold.chain_paths import chain_npz_path


def _load_filter_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_msa_row_filter.py"
    spec = importlib.util.spec_from_file_location("build_msa_row_filter_for_tests", script_path)
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
    np.savez_compressed(path, msa=msa)


def test_parse_excluded_hashes_from_mmseqs_targets(tmp_path: Path) -> None:
    module = _load_filter_module()
    digest = "a" * 64
    m8 = tmp_path / "hits.m8"
    m8.write_text(f"heldout_1\tseq_{digest}\t40.0\t80\t0.8\t0.8\t1e-6\t50\n")

    assert module._parse_excluded_hashes(m8) == {digest}


def test_scan_processed_msa_rows_skips_query_rows(tmp_path: Path) -> None:
    module = _load_filter_module()
    features_dir = tmp_path / "features"
    manifest = tmp_path / "train.txt"
    manifest.write_text("trn1_A\n")
    _write_feature(
        features_dir,
        "trn1_A",
        ["AAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCC", "CCCCCCCCCCCCCCCCCCCC"],
    )

    sequences_by_hash, stats = module._scan_processed_msa_rows(
        manifest_paths=[manifest],
        features_dirs=[features_dir],
    )

    assert stats["chains"] == 1
    assert stats["rows"] == 3
    assert stats["non_query_rows"] == 2
    assert stats["unique_non_query_rows"] == 1
    assert list(sequences_by_hash.values()) == ["CCCCCCCCCCCCCCCCCCCC"]


def test_scan_processed_msa_rows_searches_multiple_roots(tmp_path: Path) -> None:
    module = _load_filter_module()
    first_features_dir = tmp_path / "features_a"
    second_features_dir = tmp_path / "features_b"
    manifest = tmp_path / "train.txt"
    manifest.write_text("trn1_A\ntrn2_A\n")
    _write_feature(first_features_dir, "trn1_A", ["AAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCC"])
    _write_feature(second_features_dir, "trn2_A", ["DDDDDDDDDDDDDDDDDDDD", "EEEEEEEEEEEEEEEEEEEE"])

    sequences_by_hash, stats = module._scan_processed_msa_rows(
        manifest_paths=[manifest],
        features_dirs=[first_features_dir, second_features_dir],
    )

    assert stats["chains"] == 2
    assert stats["missing_features"] == 0
    assert sorted(sequences_by_hash.values()) == ["CCCCCCCCCCCCCCCCCCCC", "EEEEEEEEEEEEEEEEEEEE"]


def test_write_heldout_fasta_deduplicates_without_chain_ids(tmp_path: Path) -> None:
    module = _load_filter_module()
    manifest = tmp_path / "heldout.txt"
    manifest.write_text("val1_A\nval2_A\n")
    cache = {
        "val1_A": {"seq": "AAAAAAAAAAAAAAAAAAAA"},
        "val2_A": {"seq": "AAAAAAAAAAAAAAAAAAAA"},
    }
    fasta = tmp_path / "heldout.fasta"

    count = module._write_heldout_fasta(fasta, manifest_paths=[manifest], cache=cache)

    assert count == 1
    text = fasta.read_text()
    assert ">heldout_1" in text
    assert "val1_A" not in text
    assert "val2_A" not in text
