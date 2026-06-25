from app.core.logging import logger
from app.storage.database.base import Base
from app.storage.database.session import engine


async def init_database():

    logger.info("Initializing database")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.success("Database ready")
