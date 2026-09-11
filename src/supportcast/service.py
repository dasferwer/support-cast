import hashlib
import json
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import text

from .config import settings
from .forecast import checksum, regularize

VERSION = "sandy-v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def owned(conn, series_id, user_id, lock=False):
    row = (
        (
            await conn.execute(
                text(
                    "SELECT * FROM series WHERE id=:id AND owner_id=:owner"
                    + (" FOR UPDATE" if lock else "")
                ),
                {"id": series_id, "owner": user_id},
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        raise HTTPException(404, "Series not found")
    return row


async def enqueue(conn, series_id, revision):
    records = (
        (
            await conn.execute(
                text("SELECT day,count FROM observations WHERE series_id=:id ORDER BY day"),
                {"id": series_id},
            )
        )
        .mappings()
        .all()
    )
    payload = [{"day": r["day"].isoformat(), "count": r["count"]} for r in records]
    _, _, missing, _ = regularize(payload)
    if int((~missing).sum()) < 28 or int(missing[-28:].sum()) > 14:
        raise ValueError("Need 28 observed days and at least 14 observed days in the last 28")
    job_id = uuid4()
    await conn.execute(
        text("""
        INSERT INTO forecast_jobs(id,series_id,revision,model_version,model_sha256,input,input_sha256)
        VALUES (:id,:series,:revision,:version,:model_hash,CAST(:input AS jsonb),:input_hash)
    """),
        {
            "id": job_id,
            "series": series_id,
            "revision": revision,
            "version": VERSION,
            "model_hash": checksum(Path(settings.model_dir) / VERSION / "model.json"),
            "input": json.dumps(payload),
            "input_hash": digest(payload),
        },
    )
    return {"job_id": str(job_id), "revision": revision, "status": "pending"}
