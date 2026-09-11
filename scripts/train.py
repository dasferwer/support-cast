"""Проверяем недельный прогноз на последовательных будущих неделях, без перемешивания дат."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from supportcast.forecast import checksum, features, predict, regularize

ROOT = Path(__file__).resolve().parents[1]


def read_data():
    source = json.loads((ROOT / "data/source.json").read_text())
    for file, expected in source["files_sha256"].items():
        if checksum(ROOT / "data" / file) != expected:
            raise ValueError("Source checksum mismatch")
    with (ROOT / "data/calls.csv").open() as stream:
        rows = [
            {
                "day": datetime.strptime(r["date"], "%m/%d/%y").date().isoformat(),
                "count": int(r["total"]),
            }
            for r in csv.DictReader(stream)
        ]
    assert len({r["day"] for r in rows}) == len(rows)
    return sorted(rows, key=lambda r: r["day"]), source


def historical_period(rows):
    return [r for r in rows if r["day"] <= "2014-12-31"]


def fit(rows, alpha, version="evaluation"):
    days, values, missing, _ = regularize(rows)
    holidays = [
        x.date().isoformat()
        for x in USFederalHolidayCalendar().holidays("2012-01-01", "2032-12-31")
    ]
    models = []
    for h in range(1, 8):
        indices = [i for i in range(max(27, len(days) - 365), len(days) - h) if not missing[i + h]]
        x = np.array([features(days, values, missing, i, h, holidays) for i in indices])
        y = np.log1p(values[np.array(indices) + h])
        scaler = StandardScaler().fit(x)
        estimator = Ridge(alpha=alpha).fit(scaler.transform(x), y)
        models.append(
            {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coef": estimator.coef_.tolist(),
                "intercept": float(estimator.intercept_),
                "latest_target": days[max(indices) + h].isoformat(),
                "fit_rows": len(indices),
            }
        )
    return {
        "version": version,
        "selected_algorithm": "ridge",
        "alpha": alpha,
        "fit_through": days[-1].isoformat(),
        "holidays": holidays,
        "models": models,
    }


def evaluate(rows, cutoffs, alpha):
    periods = []
    for cutoff in cutoffs:
        past, future = rows[:cutoff], rows[cutoff : cutoff + 7]
        artifact = fit(past, alpha)
        by_day = {r["day"]: r["count"] for r in future}
        assert past[-1]["day"] < future[0]["day"]
        row = {"origin": past[-1]["day"], "fit_through": artifact["fit_through"], "algorithms": {}}
        for algorithm in ["weekly", "ridge"]:
            points = predict(artifact, past, algorithm)["points"]
            samples = [
                {"day": p["day"], "actual": by_day[p["day"]], "forecast": p["forecast"]}
                for p in points
                if p["day"] in by_day
            ]
            errors = [abs(p["actual"] - p["forecast"]) for p in samples]
            row["algorithms"][algorithm] = {
                "mae": float(np.mean(errors)),
                "wape": sum(errors) / max(1, sum(p["actual"] for p in samples)),
                "samples": samples,
            }
        periods.append(row)
    return periods


def main(version):
    import re

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", version):
        raise ValueError("Invalid version")
    directory = ROOT / "models" / version
    if directory.exists():
        raise ValueError("Version already exists")
    full_rows, source = read_data()
    rows = historical_period(full_rows)
    test_start = len(rows) - 56
    validation_start = test_start - 56
    trials = []
    for alpha in [1, 10, 100]:
        periods = evaluate(rows, range(validation_start, test_start, 7), alpha)
        trials.append(
            {
                "alpha": alpha,
                "mae": float(np.mean([p["algorithms"]["ridge"]["mae"] for p in periods])),
                "periods": periods,
            }
        )
    best = min(trials, key=lambda r: r["mae"])
    baseline_mae = float(np.mean([p["algorithms"]["weekly"]["mae"] for p in best["periods"]]))
    selected = "ridge" if best["mae"] < baseline_mae else "weekly"
    test = evaluate(rows, range(test_start, len(rows), 7), best["alpha"])
    artifact = fit(rows, best["alpha"], version)
    artifact["selected_algorithm"] = selected
    artifact["interval_radius"] = {}
    for algorithm in ["weekly", "ridge"]:
        errors = [
            [abs(s["actual"] - s["forecast"]) for s in p["algorithms"][algorithm]["samples"]]
            for p in best["periods"]
        ]
        artifact["interval_radius"][algorithm] = np.quantile(
            np.array(errors), 0.9, axis=0, method="higher"
        ).tolist()
    report = {
        "version": version,
        "dataset": source,
        "rows": len(rows),
        "first_day": rows[0]["day"],
        "last_day": rows[-1]["day"],
        "validation_start": rows[validation_start]["day"],
        "test_start": rows[test_start]["day"],
        "selected_alpha": best["alpha"],
        "selected_algorithm": selected,
        "validation_trials": trials,
        "test_periods": test,
        "protocol": "Weekly rolling origins; 7 direct horizons; fit uses at most last 365 days; no future values in features",
        "historical_period_end": "2014-12-31",
        "sparse_tail_periods": evaluate(
            full_rows, range(len(full_rows) - 56, len(full_rows), 7), best["alpha"]
        ),
        "scope_note": "2014 slice evaluated separately after inspecting sparsity in the 2017 tail; tail retained as a stress evaluation using the same selected alpha",
    }
    report["test_summary"] = {}
    for algorithm in ["weekly", "ridge"]:
        samples = [s for p in test for s in p["algorithms"][algorithm]["samples"]]
        errors = [abs(s["actual"] - s["forecast"]) for s in samples]
        radii = artifact["interval_radius"][algorithm]
        report["test_summary"][algorithm] = {
            "mae": float(np.mean(errors)),
            "wape": sum(errors) / max(1, sum(s["actual"] for s in samples)),
            "interval_coverage": float(
                np.mean([error <= radii[i % 7] for i, error in enumerate(errors)])
            ),
        }
    directory.mkdir(parents=True)
    (directory / "model.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "demo-history.json").write_text(json.dumps(rows[-365:], indent=2) + "\n")
    print(
        json.dumps(
            {"version": version, "selected": selected, "test": report["test_summary"]}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="sandy-v1")
    main(parser.parse_args().version)
