from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_metric_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "make_metric_sensitivity_table.py"
    spec = importlib.util.spec_from_file_location("make_metric_sensitivity_table_for_tests", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ordering_check_tracks_direction_by_metric() -> None:
    module = _load_metric_module()
    rows = [
        {
            "run": "medium AF2 default",
            "foldscore": 0.4,
            "lddt_ca": 0.3,
            "rmsd_ca": 20.0,
            "gdt_ha_ca": 0.1,
        },
        {
            "run": "FAPE fully unclamped",
            "foldscore": 0.5,
            "lddt_ca": 0.4,
            "rmsd_ca": 10.0,
            "gdt_ha_ca": 0.2,
        },
        {
            "run": "fixed4 + unclamped + no FT",
            "foldscore": 0.6,
            "lddt_ca": 0.5,
            "rmsd_ca": 5.0,
            "gdt_ha_ca": 0.3,
        },
    ]

    check = module._ordering_check(rows)

    assert check["all_pass"] is True
    assert check["checks"]["rmsd_ca"]["passes"] is True
