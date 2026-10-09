import json
from pathlib import Path

import pytest

from scripts.evaluate_independent import (
    evaluate,
    prefix_artifact,
    read_series,
    run,
    scenario_inputs,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_use_actual_interval_bounds_and_zero_denominator():
    rows = [
        {"actual": 0, "forecast": 2, "lower": 0, "upper": 2},
        {"actual": 10, "forecast": 8, "lower": 6, "upper": 9},
    ]
    assert summarize(rows) == {"n": 2, "mae": 2, "wape": 0.4, "coverage": 0.5, "mean_width": 2.5}
    assert summarize([rows[0]])["wape"] is None


@pytest.mark.parametrize("scenario", ["original", "missing", "level_shift", "outliers"])
def test_future_mutation_cannot_change_history(scenario):
    rows = read_series(ROOT / "data/independent/day.csv")["registered"]
    cutoff = len(rows) - 49
    history, _ = scenario_inputs(rows, cutoff, len(rows) - 56, scenario)
    changed = [dict(row) for row in rows]
    for row in changed[cutoff:]:
        row["count"] += 100000
    other, _ = scenario_inputs(changed, cutoff, len(rows) - 56, scenario)
    assert history == other
    assert history[-1]["day"] < rows[cutoff]["day"]


def test_horizon_sample_counts_and_interval_artifact_unchanged():
    rows = read_series(ROOT / "data/independent/day.csv")["casual"]
    artifact = json.loads((ROOT / "models/sandy-v1/model.json").read_text())
    before = json.dumps(artifact, sort_keys=True)
    result = evaluate(rows, artifact, "missing")
    assert json.dumps(artifact, sort_keys=True) == before
    for algorithm in result.values():
        assert algorithm["overall"]["n"] == 56
        assert all(row["n"] == 8 for row in algorithm["by_horizon"].values())


def test_external_checksum_rejected(tmp_path):
    source = tmp_path / "day.csv"
    source.write_text("changed")
    with pytest.raises(ValueError, match="сумма"):
        read_series(source)


def test_retraining_and_calibration_ignore_test_labels():
    rows = read_series(ROOT / "data/independent/day.csv")["casual"]
    changed = [dict(row) for row in rows]
    for row in changed[-56:]:
        row["count"] = 1000000
    first = prefix_artifact(rows, 100)
    second = prefix_artifact(changed, 100)
    assert first == second
    assert first["fit_through"] < rows[-56]["day"]
    assert all(model["latest_target"] < rows[-56]["day"] for model in first["models"])


def test_saved_report_reproduces_horizon_metrics():
    expected = json.loads((ROOT / "docs/independent-evaluation.json").read_text())
    actual = run()
    assert actual["protocol"] == expected["protocol"]
    for name, profiles in actual["series"].items():
        for profile, scenarios in profiles.items():
            for scenario, algorithms in scenarios.items():
                for algorithm, result in algorithms.items():
                    saved = expected["series"][name][profile][scenario][algorithm]
                    assert result["overall"] == pytest.approx(saved["overall"], abs=1e-7)
                    for horizon, values in result["by_horizon"].items():
                        assert values == pytest.approx(saved["by_horizon"][horizon], abs=1e-7)
