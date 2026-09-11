import asyncio

from sqlalchemy import text

from supportcast.db import engine


async def check():
    async with engine.connect() as conn:
        assert (
            await conn.execute(
                text(
                    "SELECT seen_at>now()-interval '60 seconds' FROM worker_heartbeats WHERE name='forecaster'"
                )
            )
        ).scalar_one_or_none()


asyncio.run(check())
