import asyncio
import json
import logging
from uuid import uuid4

from sqlalchemy import text

from .config import settings
from .db import engine
from .forecast import load_artifact, predict
from .service import digest

logger = logging.getLogger("forecaster")


async def claim():
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO worker_heartbeats(name) VALUES('forecaster') ON CONFLICT(name) DO UPDATE SET seen_at=now()"
            )
        )
        await conn.execute(
            text(
                "UPDATE forecast_jobs SET status='failed',error='Attempts exhausted' WHERE status='running' AND lease_until<now() AND attempts>=3"
            )
        )
        row = (
            (
                await conn.execute(
                    text("""
            WITH candidate AS (
                SELECT id FROM forecast_jobs WHERE (status='pending' OR (status='running' AND lease_until<now()))
                    AND attempts<3 ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
            ) UPDATE forecast_jobs SET status='running',generation=:generation,attempts=attempts+1,
                lease_until=now()+(:seconds * interval '1 second')
                WHERE id IN (SELECT id FROM candidate) RETURNING *
        """),
                    {"generation": uuid4(), "seconds": settings.lease_seconds},
                )
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


async def complete(job, result):
    async with engine.begin() as conn:
        # После истечения lease другой worker мог забрать задание; его результат нельзя перезаписать.
        changed = await conn.execute(
            text("""
            UPDATE forecast_jobs SET status='ready',result=CAST(:result AS jsonb),completed_at=now(),error=NULL
            WHERE id=:id AND generation=:generation AND status='running' AND lease_until>now()
        """),
            {"id": job["id"], "generation": job["generation"], "result": json.dumps(result)},
        )
    return changed.rowcount == 1


async def process(job):
    try:
        if digest(job["input"]) != job["input_sha256"]:
            raise ValueError("Input checksum mismatch")
        artifact = load_artifact(settings.model_dir, job["model_version"], job["model_sha256"])
        if settings.worker_delay_seconds:
            await asyncio.sleep(settings.worker_delay_seconds)
        result = await asyncio.to_thread(predict, artifact, job["input"])
        result["model_version"] = job["model_version"]
        await complete(job, result)
    except Exception as error:
        logger.warning("forecast_failed job_id=%s error_type=%s", job["id"], type(error).__name__)
        async with engine.begin() as conn:
            await conn.execute(
                text("""
                UPDATE forecast_jobs SET status=CASE WHEN attempts>=3 THEN 'failed' ELSE 'pending' END,
                    error=:error,lease_until=NULL WHERE id=:id AND generation=:generation AND status='running'
            """),
                {"id": job["id"], "generation": job["generation"], "error": type(error).__name__},
            )


async def main():
    logging.basicConfig(level=logging.INFO)
    while True:
        try:
            job = await claim()
            if job:
                await process(job)
            else:
                await asyncio.sleep(0.5)
        except Exception:
            logger.exception("worker_loop_failed")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
