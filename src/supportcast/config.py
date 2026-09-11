from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+asyncpg://supportcast:supportcast@database:5432/supportcast"
    jwt_secret: str = Field(
        default="local-demo-supportcast-replace-before-deployment", min_length=32
    )
    token_minutes: int = 60
    model_dir: str = "models"
    lease_seconds: int = 15
    worker_delay_seconds: float = 0


settings = Settings()
