"""Независимая проверка неизменного артефакта; test не используется для настройки."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.train import fit
from supportcast.forecast import predict

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = "a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be"
SCENARIOS = ("original", "missing", "level_shift", "outliers")


def read_series(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("Контрольная сумма независимого ряда не совпадает")
    with path.open() as stream:
        records = list(csv.DictReader(stream))
    result = {
        field: [{"day": row["dteday"], "count": int(row[field])} for row in records]
        for field in ("casual", "registered")
    }
    for rows in result.values():
        if len(rows) != 731 or len({row["day"] for row in rows}) != 731:
            raise ValueError("Ожидается полный уникальный ряд из 731 дня")
    return result


def scenario_inputs(rows, cutoff, test_start, scenario):
    """Стресс задаётся по индексам; изменение будущего не меняет историю."""
    if scenario not in SCENARIOS:
        raise ValueError("Неизвестный сценарий")
    history = [dict(row) for row in rows[:cutoff]]
    future = [dict(row) for row in rows[cutoff : cutoff + 7]]
    if scenario == "missing":
        history = [row for i, row in enumerate(history) if i in (0, cutoff - 1) or i % 5]
    elif scenario == "level_shift":
        for i, row in enumerate(history):
            if i >= test_start:
                row["count"] *= 2
        for row in future:
            row["count"] *= 2
    elif scenario == "outliers":
        for i in (cutoff - 3, cutoff - 10):
            history[i]["count"] *= 10
    return history, future


def summarize(samples):
    errors = [abs(row["actual"] - row["forecast"]) for row in samples]
    actual_sum = sum(row["actual"] for row in samples)
    return {
        "n": len(samples),
        "mae": sum(errors) / len(errors) if errors else None,
        "wape": sum(errors) / actual_sum if actual_sum else None,
        "coverage": sum(row["lower"] <= row["actual"] <= row["upper"] for row in samples)
        / len(samples)
        if samples
        else None,
        "mean_width": sum(row["upper"] - row["lower"] for row in samples) / len(samples)
        if samples
        else None,
    }


def evaluate(rows, artifact, scenario):
    test_start = len(rows) - 56
    output = {}
    for algorithm in ("weekly", "ridge"):
        samples = []
        for cutoff in range(test_start, len(rows), 7):
            history, future = scenario_inputs(rows, cutoff, test_start, scenario)
            predicted = predict(artifact, history, algorithm)["points"]
            truth = {row["day"]: row["count"] for row in future}
            for horizon, point in enumerate(predicted, 1):
                samples.append(
                    {
                        **point,
                        "horizon": horizon,
                        "actual": truth[point["day"]],
                        "origin": history[-1]["day"],
                    }
                )
        output[algorithm] = {
            "overall": summarize(samples),
            "by_horizon": {
                str(h): summarize([row for row in samples if row["horizon"] == h])
                for h in range(1, 8)
            },
            "samples": samples,
        }
    return output


def prefix_artifact(rows, alpha):
    """Alpha фиксирован Sandy; интервалы калибруются только до внешнего test."""
    test_start = len(rows) - 56
    calibration_start = test_start - 56
    errors = {algorithm: [] for algorithm in ("weekly", "ridge")}
    for cutoff in range(calibration_start, test_start, 7):
        artifact = fit(rows[:cutoff], alpha)
        for algorithm in errors:
            points = predict(artifact, rows[:cutoff], algorithm)["points"]
            errors[algorithm].append(
                [
                    abs(row["count"] - point["forecast"])
                    for row, point in zip(rows[cutoff : cutoff + 7], points, strict=True)
                ]
            )
    artifact = fit(rows[:test_start], alpha, "external-prefix-evaluation-only")
    artifact["interval_radius"] = {
        algorithm: np.quantile(values, 0.9, axis=0, method="higher").tolist()
        for algorithm, values in errors.items()
    }
    return artifact


def run():
    artifact_path = ROOT / "models/sandy-v1/model.json"
    artifact = json.loads(artifact_path.read_text())
    series = read_series(ROOT / "data/independent/day.csv")
    return {
        "protocol": {
            "model_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "model_version": artifact["version"],
            "selected_algorithm": artifact["selected_algorithm"],
            "data_sha256": DATA_SHA,
            "test_start": next(iter(series.values()))[-56]["day"],
            "test_end": next(iter(series.values()))[-1]["day"],
            "origins": 8,
            "horizons": 7,
            "profiles": {
                "frozen_transfer": "Неизменный Sandy artifact и радиусы; ретроспективный перенос, не исторический backtest 2012",
                "prefix_retrained": "Alpha Sandy фиксирован; rolling calibration 56 дней до test; финальный fit только pre-test; test не настраивает параметры",
            },
            "intervals": "Номинальный уровень 90%; 8 калибровочных ошибок на горизонт, без гарантии coverage",
            "missing": "Каждый пятый день только истории, кроме первого/последнего",
            "level_shift": "Удвоение фактов с первого дня test; доступно только после наблюдения",
            "outliers": "Два десятикратных выброса только истории: origin-2 и origin-9",
        },
        "series": {
            name: {
                profile: {scenario: evaluate(rows, model, scenario) for scenario in SCENARIOS}
                for profile, model in (
                    ("frozen_transfer", artifact),
                    ("prefix_retrained", prefix_artifact(rows, artifact["alpha"])),
                )
            }
            for name, rows in series.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/independent-evaluation.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), ensure_ascii=False, indent=2) + "\n")
    print(args.output)
