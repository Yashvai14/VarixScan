from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./varicose_vein_app.db"  # Fallback for local development
    ENVIRONMENT: str = "development"
    FRONTEND_URL: str = "http://localhost:3001"
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379"
    RF_API_KEY: Optional[str] = None
    RF_PUBLIC_KEY: Optional[str] = None
    RF_MODEL_ID: str = "varicose-veins"
    RF_VERSION: str = "1"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
