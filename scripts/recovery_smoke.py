"""Прерываем расчёт и проверяем, что новый worker завершит то же задание после lease."""

import json
import os
import subprocess
import time
from pathlib import Path

from smoke import setup, wait_job

ROOT = Path(__file__).resolve().parents[1]


def compose(delay, *args):
    subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        env={**os.environ, "WORKER_DELAY_SECONDS": str(delay)},
        capture_output=True,
        check=True,
    )


def main():
    try:
        compose(30, "up", "-d", "--no-deps", "--force-recreate", "worker")
        client, series, history = setup("http://localhost:8170")
        result, _ = client.request(
            "PUT",
            f"/series/{series['id']}/observations",
            {"observations": history},
            {"Idempotency-Key": "recovery", "If-Match": '"0"'},
            expected=202,
        )
        for _ in range(50):
            job, _ = client.request("GET", f"/jobs/{result['job_id']}")
            if job["status"] == "running":
                break
            time.sleep(0.2)
        assert job["status"] == "running"
        compose(30, "kill", "--signal", "SIGKILL", "worker")
    finally:
        compose(0, "up", "-d", "--no-deps", "--force-recreate", "worker")
    job = wait_job(client, result["job_id"])
    assert job["attempts"] == 2 and job["status"] == "ready"
    print(
        json.dumps(
            {
                "ok": True,
                "fault": "SIGKILL forecaster",
                "attempts": 2,
                "same_job_id": job["id"] == result["job_id"],
                "revision": job["revision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
