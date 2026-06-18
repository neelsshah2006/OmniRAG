from fastapi import FastAPI
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
)


@app.get("/health")
async def health():
    return {
        "status": "running",
        "environment": settings.ENVIRONMENT,
    }
