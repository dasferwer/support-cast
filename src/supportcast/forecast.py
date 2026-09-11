import hashlib
import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regularize(observations):
    if not observations:
        raise ValueError("At least 56 calendar days are required")
    known = {date.fromisoformat(r["day"]): r["count"] for r in observations}
    first, last = min(known), max(known)
    size = (last - first).days + 1
    if not 56 <= size <= 3650:
        raise ValueError("History must cover 56 to 3650 days")
    days, values, missing, anomalies = [], [], [], []
    for index in range(size):
        day = first + timedelta(days=index)
        value = known.get(day)
        absent = value is None
        if absent:
            # Пропуск заполняем только уже известными днями. Будущее не участвует даже в очистке.
            weekly = [values[j] for j in range(index - 7, max(-1, index - 29), -7)]
            value = float(np.median(weekly or values[-28:] or [0]))
        previous = np.array(values[-28:])
        median = float(np.median(previous)) if len(previous) else float(value)
        mad = float(np.median(np.abs(previous - median))) if len(previous) else 0
        anomaly = (
            not absent
            and len(previous) >= 14
            and abs(value - median) > max(6 * mad, 3 * median, 20)
        )
        days.append(day)
        values.append(float(value))
        missing.append(absent)
        anomalies.append(anomaly)
    return days, np.array(values), np.array(missing), np.array(anomalies)


def features(days, values, missing, origin, horizon, holidays):
    target = days[origin] + timedelta(days=horizon)
    past = values[origin - 27 : origin + 1]
    # Ограничиваем влияние старого всплеска на признаки, сохраняя исходные факты для отчёта.
    cap = max(float(np.median(past)) * 5, 20)
    lagged = np.log1p(np.minimum(past[::-1], cap))
    return np.array(
        [
            *lagged,
            np.log1p(np.mean(past[-7:])),
            np.log1p(np.mean(past)),
            *[float(target.weekday() == i) for i in range(7)],
            np.sin(2 * np.pi * target.timetuple().tm_yday / 365.25),
            np.cos(2 * np.pi * target.timetuple().tm_yday / 365.25),
            float(target.isoformat() in holidays),
            float(np.mean(missing[origin - 27 : origin + 1])),
        ]
    )


def weekly(values, origin, horizon):
    return float(values[origin + horizon - 7])


def predict(artifact, observations, algorithm=None):
    days, values, missing, anomalies = regularize(observations)
    origin = len(days) - 1
    chosen = algorithm or artifact["selected_algorithm"]
    points = []
    for horizon in range(1, 8):
        baseline = weekly(values, origin, horizon)
        if chosen == "weekly":
            estimate = baseline
        else:
            model = artifact["models"][horizon - 1]
            x = features(days, values, missing, origin, horizon, artifact["holidays"])
            log_estimate = ((x - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(
                model["coef"]
            ) + model["intercept"]
            estimate = max(0, float(np.expm1(np.clip(log_estimate, 0, np.log1p(1000000)))))
        radius = artifact.get("interval_radius", {}).get(chosen, [0] * 7)[horizon - 1]
        points.append(
            {
                "day": (days[-1] + timedelta(days=horizon)).isoformat(),
                "forecast": round(estimate, 2),
                "weekly_baseline": baseline,
                "lower": round(max(0, estimate - radius), 2),
                "upper": round(estimate + radius, 2),
            }
        )
    return {
        "algorithm": chosen,
        "origin": days[-1].isoformat(),
        "points": points,
        "quality": {
            "missing_days": int(missing.sum()),
            "anomaly_days": int(anomalies.sum()),
            "missing_last_28": int(missing[-28:].sum()),
        },
    }


@lru_cache(maxsize=2)
def load_artifact(root, version, expected):
    import re

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", version):
        raise ValueError("Invalid version")
    path = Path(root) / version / "model.json"
    if checksum(path) != expected:
        raise ValueError("Model checksum mismatch")
    artifact = json.loads(path.read_text())
    if artifact["version"] != version or len(artifact["models"]) != 7:
        raise ValueError("Invalid model")
    return artifact
