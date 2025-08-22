"""
Client Manager Service

Manages database and API client connections.
"""

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..config.logfire_config import search_logger
from ..config.mongodb_config import get_mongodb_database, test_mongodb_connection


async def get_mongodb_client() -> AsyncIOMotorDatabase:
    """
    Get a MongoDB database instance.

    Returns:
        MongoDB database instance
    """
    try:
        db = get_mongodb_database()

        # Test connection
        connection_ok = await test_mongodb_connection()
        if not connection_ok:
            raise ConnectionError("Failed to connect to MongoDB")

        search_logger.info("MongoDB client initialized successfully")
        return db
    except Exception as e:
        search_logger.error(f"Failed to create MongoDB client: {e}")
        raise
