"""
MongoDB configuration and connection management.
"""

import os
from dataclasses import dataclass
from urllib.parse import quote_plus

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient


class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""
    pass


@dataclass
class MongoDBConfig:
    """Configuration for MongoDB connection."""
    
    connection_string: str
    database_name: str
    host: str = "localhost"
    port: int = 27017
    username: str | None = None
    password: str | None = None
    auth_source: str = "admin"
    tls: bool = False
    

def validate_mongodb_config(config: MongoDBConfig) -> bool:
    """Validate MongoDB configuration."""
    if not config.database_name:
        raise ConfigurationError("MongoDB database name cannot be empty")
    
    if config.connection_string:
        # Using connection string - minimal validation
        if not config.connection_string.startswith(("mongodb://", "mongodb+srv://")):
            raise ConfigurationError("MongoDB connection string must start with 'mongodb://' or 'mongodb+srv://'")
    else:
        # Using individual parameters
        if not config.host:
            raise ConfigurationError("MongoDB host cannot be empty")
        if config.port <= 0 or config.port > 65535:
            raise ConfigurationError(f"MongoDB port must be between 1-65535, got: {config.port}")
    
    return True


def build_connection_string(config: MongoDBConfig) -> str:
    """Build MongoDB connection string from config."""
    if config.connection_string:
        return config.connection_string
    
    # Build connection string from components
    protocol = "mongodb+srv" if config.tls else "mongodb"
    auth_part = ""
    
    if config.username and config.password:
        # URL encode username and password to handle special characters
        username = quote_plus(config.username)
        password = quote_plus(config.password)
        auth_part = f"{username}:{password}@"
    
    host_part = f"{config.host}:{config.port}" if not config.tls else config.host
    auth_source_part = f"?authSource={config.auth_source}" if config.username else ""
    
    return f"{protocol}://{auth_part}{host_part}/{config.database_name}{auth_source_part}"


def load_mongodb_config() -> MongoDBConfig:
    """Load MongoDB configuration from environment variables."""
    # Primary: Use connection string if provided
    connection_string = os.getenv("MONGODB_CONNECTION_STRING") or os.getenv("MONGODB_URI")
    
    # Database name (required)
    database_name = os.getenv("MONGODB_DATABASE", "archon")
    
    if connection_string:
        config = MongoDBConfig(
            connection_string=connection_string,
            database_name=database_name
        )
    else:
        # Build from individual components
        host = os.getenv("MONGODB_HOST", "localhost")
        port = int(os.getenv("MONGODB_PORT", "27017"))
        username = os.getenv("MONGODB_USERNAME")
        password = os.getenv("MONGODB_PASSWORD")
        auth_source = os.getenv("MONGODB_AUTH_SOURCE", "admin")
        tls = os.getenv("MONGODB_TLS", "false").lower() in ("true", "1", "yes", "on")
        
        config = MongoDBConfig(
            connection_string="",
            database_name=database_name,
            host=host,
            port=port,
            username=username,
            password=password,
            auth_source=auth_source,
            tls=tls
        )
    
    validate_mongodb_config(config)
    return config


def get_mongodb_config() -> MongoDBConfig:
    """Get MongoDB configuration with validation."""
    return load_mongodb_config()


# Global client instances (lazy initialized)
_sync_client: MongoClient | None = None
_async_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None


def get_sync_mongodb_client() -> MongoClient:
    """Get synchronous MongoDB client."""
    global _sync_client
    if _sync_client is None:
        config = get_mongodb_config()
        connection_string = build_connection_string(config)
        _sync_client = MongoClient(connection_string)
    return _sync_client


def get_async_mongodb_client() -> AsyncIOMotorClient:
    """Get asynchronous MongoDB client."""
    global _async_client
    if _async_client is None:
        config = get_mongodb_config()
        connection_string = build_connection_string(config)
        _async_client = AsyncIOMotorClient(connection_string)
    return _async_client


def get_mongodb_database() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance."""
    global _database
    if _database is None:
        config = get_mongodb_config()
        client = get_async_mongodb_client()
        _database = client[config.database_name]
    return _database


async def test_mongodb_connection() -> bool:
    """Test MongoDB connection."""
    try:
        db = get_mongodb_database()
        # Simple ping to test connection
        await db.command("ping")
        return True
    except Exception:
        return False


async def initialize_mongodb_collections():
    """Initialize MongoDB collections with indexes."""
    db = get_mongodb_database()
    
    # Sources collection
    sources = db.sources
    await sources.create_index("source_id", unique=True)
    await sources.create_index("created_at")
    await sources.create_index("source_type")
    
    # Documents collection with vector search index
    documents = db.documents
    await documents.create_index("source_id")
    await documents.create_index("created_at")
    await documents.create_index([("embedding", "2dsphere")])  # For vector similarity
    
    # Projects collection
    projects = db.projects
    await projects.create_index("project_id", unique=True)
    await projects.create_index("created_at")
    await projects.create_index("status")
    
    # Tasks collection
    tasks = db.tasks
    await tasks.create_index("task_id", unique=True)
    await tasks.create_index("project_id")
    await tasks.create_index("created_at")
    await tasks.create_index("status")
    
    # Code examples collection
    code_examples = db.code_examples
    await code_examples.create_index("source_id")
    await code_examples.create_index("language")
    await code_examples.create_index("created_at")


def close_mongodb_connections():
    """Close MongoDB connections."""
    global _sync_client, _async_client, _database
    
    if _sync_client:
        _sync_client.close()
        _sync_client = None
    
    if _async_client:
        _async_client.close()
        _async_client = None
    
    _database = None