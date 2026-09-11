import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import httpx
import jwt
import pytest
import pytest_asyncio
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supportcast.config import settings  # noqa: E402
from supportcast.db import engine  # noqa: E402
from supportcast.main import app  # noqa: E402

USER = UUID("17000000-0000-0000-0000-000000000101")
OTHER = UUID("17000000-0000-0000-0000-000000000102")


def headers(user=USER):
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": str(user),
            "iat": now,
            "exp": now + timedelta(minutes=10),
            "iss": "supportcast",
            "aud": "supportcast",
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    return {"Authorization": "Bearer " + token}


@pytest.fixture(scope="session")
def history():
    return json.loads((ROOT / "models/sandy-v1/demo-history.json").read_text())


@pytest_asyncio.fixture(autouse=True)
async def database():
    if os.environ.get("TESTING") != "true" or not settings.database_url.endswith(
        "/supportcast_test"
    ):
        raise RuntimeError("Isolated test database required")
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE users,series,observations,forecast_jobs,ingestion_requests,worker_heartbeats CASCADE"
            )
        )
        for user in [USER, OTHER]:
            await conn.execute(
                text("INSERT INTO users VALUES(:id,:email,'unused','user')"),
                {"id": user, "email": str(user) + "@example.com"},
            )


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers()
    ) as c:
        yield c


@pytest_asyncio.fixture
async def series(client):
    return (await client.post("/series", json={"name": "Support"})).json()["id"]
