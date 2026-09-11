import asyncio
import json
from datetime import date

import numpy as np
import pytest
from conftest import OTHER, ROOT, headers
from sqlalchemy import text

from scripts.train import evaluate, fit, historical_period, read_data
from supportcast.db import engine
from supportcast.forecast import features, load_artifact, predict, regularize
from supportcast.worker import claim, complete, process


async def ingest(client, series, history, key="batch-1", version=0):
    return await client.put(
        f"/series/{series}/observations",
        json={"observations": history},
        headers={"Idempotency-Key": key, "If-Match": f'"{version}"'},
    )


async def test_concurrent_retries_create_one_snapshot(client, series, history):
    responses = await asyncio.gather(*(ingest(client, series, history) for _ in range(10)))
    assert sorted(r.status_code for r in responses) == [200] * 9 + [202]
    assert len({r.json()["job_id"] for r in responses}) == 1
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM forecast_jobs"))).scalar_one() == 1


async def test_conflict_and_stale_etag_preserve_data(client, series, history):
    await ingest(client, series, history)
    changed = [{**history[-1], "count": 123}]
    assert (await ingest(client, series, changed)).status_code == 409
    assert (await ingest(client, series, changed, key="next")).status_code == 412
    assert (await client.get(f"/series/{series}")).json()["revision"] == 1


async def test_concurrent_edits_allow_only_one_current_revision(client, series, history):
    await ingest(client, series, history)
    results = await asyncio.gather(
        *(
            ingest(client, series, [{**history[-1], "count": i}], key=f"key-{i}", version=1)
            for i in range(8)
        )
    )
    assert sorted(r.status_code for r in results) == [202] + [412] * 7


async def test_tenant_isolation_for_series_and_jobs(client, series, history):
    result = await ingest(client, series, history)
    for path in [
        f"/series/{series}",
        f"/jobs/{result.json()['job_id']}",
        f"/series/{series}/forecast",
    ]:
        assert (await client.get(path, headers=headers(OTHER))).status_code == 404
    assert (await client.get("/series", headers=headers(OTHER))).json() == []


async def test_invalid_batch_rolls_back_all_observations(client, series, history):
    assert (await ingest(client, series, history[:5])).status_code == 422
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM observations"))).scalar_one() == 0
    assert (await client.get(f"/series/{series}")).json()["revision"] == 0
    assert (await ingest(client, series, [history[0], history[0]])).status_code == 422
    assert (
        await ingest(client, series, [{"day": date.today().isoformat(), "count": 1}])
    ).status_code == 422


async def test_missing_etag_rejected(client, series, history):
    response = await client.put(
        f"/series/{series}/observations",
        headers={"Idempotency-Key": "a"},
        json={"observations": history},
    )
    assert response.status_code == 428


async def test_worker_commits_forecast_and_staffing(client, series, history):
    response = await ingest(client, series, history)
    job = await claim()
    await process(job)
    result = (await client.get(f"/jobs/{response.json()['job_id']}")).json()
    assert result["status"] == "ready" and len(result["result"]["points"]) == 7
    assert not result["stale"]
    assert (await client.get(f"/jobs/{job['id']}/staffing")).status_code == 200


async def test_expired_generation_cannot_overwrite_recovered_job(client, series, history):
    await ingest(client, series, history)
    old = await claim()
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE forecast_jobs SET lease_until=now()-interval '1 second'"))
    recovered = await claim()
    assert recovered["generation"] != old["generation"] and recovered["attempts"] == 2
    assert not await complete(old, {"wrong": True})
    await process(recovered)
    assert not await complete(old, {"wrong": True})


async def test_two_workers_claim_only_once(client, series, history):
    await ingest(client, series, history)
    results = await asyncio.gather(claim(), claim())
    assert sum(r is not None for r in results) == 1


async def test_old_forecast_is_explicitly_stale(client, series, history):
    first = (await ingest(client, series, history)).json()
    await process(await claim())
    await ingest(client, series, [{**history[-1], "count": 50}], key="edit", version=1)
    assert (await client.get(f"/jobs/{first['job_id']}")).json()["stale"]
    assert (await client.get(f"/series/{series}/forecast")).json()["revision"] == 2


async def test_snapshot_survives_later_observation_edits(client, series, history):
    await ingest(client, series, history)
    job = await claim()
    await ingest(client, series, [{**history[-1], "count": 98765}], key="edit", version=1)
    assert job["input"][-1]["count"] == history[-1]["count"]
    await process(job)


async def test_corrupted_snapshot_exhausts_three_attempts(client, series, history):
    await ingest(client, series, history)
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE forecast_jobs SET input_sha256='broken'"))
    for _ in range(3):
        await process(await claim())
    assert await claim() is None
    assert (await client.get(f"/series/{series}/forecast")).json()["status"] == "failed"


async def test_seed_is_idempotent():
    from scripts.seed import seed

    await seed()
    await seed()
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM forecast_jobs"))).scalar_one() == 1


def test_missing_values_are_filled_only_from_the_past(history):
    rows = [dict(r) for r in history]
    rows[60]["count"] = None
    before = regularize(rows)[1][60]
    rows[61]["count"] = 1000000
    assert regularize(rows)[1][60] == before


def test_future_values_do_not_change_origin_features(history):
    days, values, missing, _ = regularize(history)
    original = features(days, values, missing, 80, 7, [])
    values[81:] = 1000000
    assert np.array_equal(original, features(days, values, missing, 80, 7, []))


def test_weekly_baseline_uses_last_known_week(history):
    artifact = json.loads((ROOT / "models/sandy-v1/model.json").read_text())
    result = predict(artifact, history, "weekly")
    assert [p["forecast"] for p in result["points"]] == [r["count"] for r in history[-7:]]


def test_model_file_integrity(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken/model.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(str(tmp_path), "broken", "wrong")
    with pytest.raises(ValueError, match="Invalid version"):
        load_artifact(str(tmp_path), "../broken", "wrong")


def test_training_targets_never_cross_cutoff(history):
    artifact = fit(history[:200], 10)
    assert all(m["latest_target"] <= history[199]["day"] for m in artifact["models"])


def test_recompute_rolling_evaluation_from_original_data():
    full, _ = read_data()
    rows = historical_period(full)
    report = json.loads((ROOT / "models/sandy-v1/evaluation.json").read_text())
    periods = evaluate(rows, range(len(rows) - 56, len(rows), 7), report["selected_alpha"])
    for actual, expected in zip(periods, report["test_periods"], strict=True):
        assert actual["origin"] == expected["origin"]
        for algorithm in ["weekly", "ridge"]:
            assert actual["algorithms"][algorithm]["mae"] == pytest.approx(
                expected["algorithms"][algorithm]["mae"], abs=1e-9
            )


def test_extreme_spike_is_reported_without_destroying_source(history):
    rows = [dict(r) for r in history]
    rows[-1]["count"] = 999999
    _, values, _, anomalies = regularize(rows)
    assert values[-1] == 999999 and anomalies[-1]
