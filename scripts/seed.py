"""Добавляем деморяд только при первом запуске, сохраняя последующие правки пользователя."""

import asyncio
import json
from datetime import date
from pathlib import Path
from uuid import UUID

from sqlalchemy import text

from supportcast.auth import hasher
from supportcast.config import settings
from supportcast.db import engine
from supportcast.service import VERSION, enqueue

DEMO = UUID("17000000-0000-0000-0000-000000000001")
SERIES = UUID("17000000-0000-0000-0000-000000000010")


async def seed():
    async with engine.begin() as conn:
        await conn.execute(text("SELECT pg_advisory_xact_lock(170001)"))
        await conn.execute(
            text(
                "INSERT INTO users(id,email,password_hash,role) VALUES(:id,'demo@example.com',:hash,'admin') ON CONFLICT DO NOTHING"
            ),
            {"id": DEMO, "hash": hasher.hash("SupportCastDemo123!")},
        )
        exists = (
            await conn.execute(text("SELECT id FROM series WHERE id=:id"), {"id": SERIES})
        ).first()
        if exists:
            return
        await conn.execute(
            text(
                "INSERT INTO series(id,owner_id,name,revision) VALUES(:id,:owner,'NYC 311 Sandy-related calls',1)"
            ),
            {"id": SERIES, "owner": DEMO},
        )
        rows = json.loads((Path(settings.model_dir) / VERSION / "demo-history.json").read_text())
        await conn.execute(
            text("INSERT INTO observations VALUES(:id,:day,:count)"),
            [
                {"id": SERIES, "day": date.fromisoformat(r["day"]), "count": r["count"]}
                for r in rows
            ],
        )
        await enqueue(conn, SERIES, 1)
    print("Demo series ready")


if __name__ == "__main__":
    asyncio.run(seed())
