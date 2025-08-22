"""
Environment configuration management for the MCP server.
"""

import os
from dataclasses import dataclass


class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""

    pass


@dataclass
class EnvironmentConfig:
    """Configuration loaded from environment variables."""

    port: int  # Required - no default
    mongodb_connection_string: str | None = None
    mongodb_database: str = "archon"
    openai_api_key: str | None = None
    host: str = "0.0.0.0"
    transport: str = "sse"


@dataclass
class RAGStrategyConfig:
    """Configuration for RAG strategies."""

    use_contextual_embeddings: bool = False
    use_hybrid_search: bool = True
    use_agentic_rag: bool = True
    use_reranking: bool = True


def validate_openai_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    if not api_key:
        raise ConfigurationError("OpenAI API key cannot be empty")

    if not api_key.startswith("sk-"):
        raise ConfigurationError("OpenAI API key must start with 'sk-'")

    return True


def validate_mongodb_connection_string(connection_string: str) -> bool:
    """Validate MongoDB connection string format."""
    if not connection_string:
        raise ConfigurationError("MongoDB connection string cannot be empty")

    if not connection_string.startswith(("mongodb://", "mongodb+srv://")):
        raise ConfigurationError("MongoDB connection string must start with 'mongodb://' or 'mongodb+srv://'")

    return True


def load_environment_config() -> EnvironmentConfig:
    """Load and validate environment configuration."""
    # OpenAI API key is optional at startup - can be set via API
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # MongoDB connection configuration
    mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING") or os.getenv("MONGODB_URI")
    mongodb_database = os.getenv("MONGODB_DATABASE", "archon")

    # Validate MongoDB connection if provided
    if mongodb_connection_string:
        validate_mongodb_connection_string(mongodb_connection_string)

    # Validate OpenAI API key if provided
    if openai_api_key:
        validate_openai_api_key(openai_api_key)

    # Optional environment variables with defaults
    host = os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("PORT")
    if not port_str:
        # This appears to be for MCP configuration based on default 8051
        port_str = os.getenv("ARCHON_MCP_PORT")
        if not port_str:
            raise ConfigurationError(
                "PORT or ARCHON_MCP_PORT environment variable is required. "
                "Please set it in your .env file or environment. "
                "Default value: 8051"
            )
    transport = os.getenv("TRANSPORT", "sse")

    # Validate and convert port
    try:
        port = int(port_str)
    except ValueError as e:
        raise ConfigurationError(f"PORT must be a valid integer, got: {port_str}") from e

    return EnvironmentConfig(
        openai_api_key=openai_api_key,
        mongodb_connection_string=mongodb_connection_string,
        mongodb_database=mongodb_database,
        host=host,
        port=port,
        transport=transport,
    )


def get_config() -> EnvironmentConfig:
    """Get environment configuration with validation."""
    return load_environment_config()


def get_rag_strategy_config() -> RAGStrategyConfig:
    """Load RAG strategy configuration from environment variables."""

    def str_to_bool(value: str | None) -> bool:
        """Convert string environment variable to boolean."""
        if value is None:
            return False
        return value.lower() in ("true", "1", "yes", "on")

    return RAGStrategyConfig(
        use_contextual_embeddings=str_to_bool(os.getenv("USE_CONTEXTUAL_EMBEDDINGS")),
        use_hybrid_search=str_to_bool(os.getenv("USE_HYBRID_SEARCH")),
        use_agentic_rag=str_to_bool(os.getenv("USE_AGENTIC_RAG")),
        use_reranking=str_to_bool(os.getenv("USE_RERANKING")),
    )
