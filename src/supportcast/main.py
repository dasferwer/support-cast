import json
import math
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from .auth import User, current_user, router
from .config import settings
from .db import engine
from .observability import instrument
from .service import VERSION, digest, enqueue, owned

app = FastAPI(title="SupportCast", version="0.1.0")
app.include_router(router)
instrument(app)


class SeriesInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class Observation(BaseModel):
    day: date
    count: int | None = Field(ge=0, le=1000000)

    @field_validator("day")
    @classmethod
    def past_day(cls, value):
        if value >= date.today():
            raise ValueError("Only completed calendar days are accepted")
        return value


class Batch(BaseModel):
    observations: list[Observation] = Field(min_length=1, max_length=3650)

    @field_validator("observations")
    @classmethod
    def unique_days(cls, values):
        if len({v.day for v in values}) != len(values):
            raise ValueError("A batch cannot contain duplicate dates")
        return values


@app.get("/health")
async def health():
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return {"status": "ok", "model_version": VERSION}


@app.post("/series", status_code=201)
async def create_series(data: SeriesInput, user: User = Depends(current_user)):
    async with engine.begin() as conn:
        await conn.execute(text("SELECT id FROM users WHERE id=:id FOR UPDATE"), {"id": user.id})
        count = (
            await conn.execute(
                text("SELECT count(*) FROM series WHERE owner_id=:id"), {"id": user.id}
            )
        ).scalar_one()
        if count >= 20:
            raise HTTPException(409, "Maximum 20 series per user")
        row = (
            (
                await conn.execute(
                    text(
                        "INSERT INTO series(id,owner_id,name) VALUES(:id,:owner,:name) RETURNING *"
                    ),
                    {"id": uuid4(), "owner": user.id, "name": data.name},
                )
            )
            .mappings()
            .one()
        )
    return dict(row)


@app.get("/series")
async def list_series(user: User = Depends(current_user)):
    async with engine.connect() as conn:
        return [
            dict(r)
            for r in (
                await conn.execute(
                    text("SELECT * FROM series WHERE owner_id=:id ORDER BY created_at"),
                    {"id": user.id},
                )
            ).mappings()
        ]


@app.get("/series/{series_id}")
async def get_series(series_id: UUID, response: Response, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        row = await owned(conn, series_id, user.id)
    response.headers["ETag"] = f'"{row["revision"]}"'
    return dict(row)


@app.put("/series/{series_id}/observations", status_code=202)
async def ingest(
    series_id: UUID,
    data: Batch,
    response: Response,
    idempotency_key: str = Header(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"),
    if_match: str | None = Header(default=None),
    user: User = Depends(current_user),
):
    payload = sorted([r.model_dump(mode="json") for r in data.observations], key=lambda r: r["day"])
    body_hash = digest(payload)
    async with engine.begin() as conn:
        series = await owned(conn, series_id, user.id, True)
        previous = (
            (
                await conn.execute(
                    text(
                        "SELECT body_hash,result FROM ingestion_requests WHERE series_id=:id AND request_key=:key"
                    ),
                    {"id": series_id, "key": idempotency_key},
                )
            )
            .mappings()
            .first()
        )
        if previous:
            if previous["body_hash"] != body_hash:
                raise HTTPException(409, "Key already used with different data")
            response.status_code = 200
            return {**previous["result"], "replayed": True}
        if if_match is None:
            raise HTTPException(428, "If-Match is required")
        if if_match != f'"{series["revision"]}"':
            raise HTTPException(412, "Series changed; read its ETag again")
        await conn.execute(
            text("""
            INSERT INTO observations(series_id,day,count) VALUES(:series,:day,:count)
            ON CONFLICT(series_id,day) DO UPDATE SET count=EXCLUDED.count
        """),
            [{"series": series_id, "day": r.day, "count": r.count} for r in data.observations],
        )
        revision = series["revision"] + 1
        try:
            result = await enqueue(conn, series_id, revision)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        # Данные и задание прогноза коммитятся вместе, поэтому принятая загрузка не теряется.
        await conn.execute(
            text("UPDATE series SET revision=:v WHERE id=:id"), {"v": revision, "id": series_id}
        )
        await conn.execute(
            text("INSERT INTO ingestion_requests VALUES(:id,:key,:hash,CAST(:result AS jsonb))"),
            {
                "id": series_id,
                "key": idempotency_key,
                "hash": body_hash,
                "result": json.dumps(result),
            },
        )
    response.headers["ETag"] = f'"{revision}"'
    return {**result, "replayed": False}


@app.get("/jobs/{job_id}")
async def get_job(job_id: UUID, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    text("""
            SELECT j.id,j.series_id,j.revision,j.status,j.attempts,j.result,j.error,j.input_sha256,j.model_version,
                   j.revision!=s.revision AS stale
            FROM forecast_jobs j JOIN series s ON s.id=j.series_id WHERE j.id=:id AND s.owner_id=:owner
        """),
                    {"id": job_id, "owner": user.id},
                )
            )
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(404, "Job not found")
    return dict(row)


@app.get("/series/{series_id}/forecast")
async def latest(series_id: UUID, user: User = Depends(current_user)):
    async with engine.connect() as conn:
        await owned(conn, series_id, user.id)
        job = (
            await conn.execute(
                text(
                    "SELECT id FROM forecast_jobs WHERE series_id=:id ORDER BY revision DESC LIMIT 1"
                ),
                {"id": series_id},
            )
        ).scalar_one_or_none()
    if job is None:
        raise HTTPException(404, "Upload at least 56 days of observations first")
    return await get_job(job, user)


@app.get("/jobs/{job_id}/staffing")
async def staffing(
    job_id: UUID,
    handling_minutes: float = Query(default=8, gt=0, le=60),
    productive_hours: float = Query(default=6, gt=0, le=12),
    user: User = Depends(current_user),
):
    job = await get_job(job_id, user)
    if job["status"] != "ready":
        raise HTTPException(409, "Forecast is not ready")
    capacity = productive_hours * 60 / handling_minutes
    return {
        "method": "Daily workload divided by operator productive capacity; not an SLA queueing model",
        "points": [
            {
                "day": p["day"],
                "operators": math.ceil(p["forecast"] / capacity),
                "upper_scenario_operators": math.ceil(p["upper"] / capacity),
            }
            for p in job["result"]["points"]
        ],
    }


@app.get("/model")
async def model_report(user: User = Depends(current_user)):
    return json.loads((Path(settings.model_dir) / VERSION / "evaluation.json").read_text())
