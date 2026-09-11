"""Загружаем историю, дожидаемся прогноза и проверяем автоматическое обновление после правки."""

import json
import time
from pathlib import Path
from uuid import uuid4

from http_client import Client


def wait_job(client, job_id):
    for _ in range(100):
        job, _ = client.request("GET", f"/jobs/{job_id}")
        if job["status"] == "ready":
            return job
        assert job["status"] != "failed", job
        time.sleep(0.3)
    raise AssertionError("Forecast did not complete")


def setup(base="http://localhost:8000"):
    client = Client(base)
    email = f"smoke-{uuid4().hex}@example.com"
    client.request(
        "POST", "/auth/register", {"email": email, "password": "SupportCastDemo123!"}, expected=201
    )
    client.login(email)
    series, _ = client.request("POST", "/series", {"name": "Smoke forecast"}, expected=201)
    history = json.loads(
        (Path(__file__).resolve().parents[1] / "models/sandy-v1/demo-history.json").read_text()
    )
    return client, series, history


def main():
    client, series, history = setup()
    path = f"/series/{series['id']}/observations"
    headers = {"Idempotency-Key": "first", "If-Match": '"0"'}
    result, _ = client.request("PUT", path, {"observations": history}, headers, expected=202)
    replay, _ = client.request("PUT", path, {"observations": history}, headers)
    assert replay["replayed"] and result["job_id"] == replay["job_id"]
    first = wait_job(client, result["job_id"])
    changed = [{**history[-1], "count": history[-1]["count"] + 25}]
    new, _ = client.request(
        "PUT",
        path,
        {"observations": changed},
        {"Idempotency-Key": "edit", "If-Match": '"1"'},
        expected=202,
    )
    second = wait_job(client, new["job_id"])
    old, _ = client.request("GET", f"/jobs/{first['id']}")
    assert old["stale"] and second["revision"] == 2 and not second["stale"]
    staff, _ = client.request("GET", f"/jobs/{second['id']}/staffing")
    assert len(staff["points"]) == 7
    print(
        json.dumps(
            {
                "ok": True,
                "horizon": 7,
                "automatic_refresh": True,
                "idempotent_ingestion": True,
                "immutable_snapshot": first["input_sha256"] != second["input_sha256"],
                "revision": 2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
