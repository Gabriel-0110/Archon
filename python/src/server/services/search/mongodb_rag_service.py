"""
MongoDB RAG Service - Coordinator for MongoDB-based search strategies

This service acts as a coordinator that delegates to MongoDB-based strategy implementations.
It combines multiple RAG strategies in a pipeline fashion using MongoDB as the backend.
"""

import asyncio
import os
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...config.logfire_config import get_logger, safe_span
from ...config.mongodb_config import get_mongodb_database
from ..embeddings.embedding_service import create_embedding
from .mongodb_search_strategies import MongoDBBaseSearchStrategy, MongoDBHybridSearchStrategy

logger = get_logger(__name__)


class MongoDBRAGService:
    """
    Coordinator service that orchestrates multiple RAG strategies using MongoDB.

    This service delegates to MongoDB-based strategy implementations and combines them
    based on configuration settings.
    """

    def __init__(self, db: AsyncIOMotorDatabase = None):
        """Initialize RAG service as a coordinator for MongoDB search strategies"""
        self.db = db or get_mongodb_database()

        # Initialize base strategy (always needed)
        self.base_strategy = MongoDBBaseSearchStrategy(self.db)

        # Initialize hybrid strategy
        self.hybrid_strategy = MongoDBHybridSearchStrategy(self.db)

        # Initialize vector search indexes
        asyncio.create_task(self._ensure_search_indexes())

    async def _ensure_search_indexes(self):
        """Ensure search indexes are created."""
        try:
            await self.base_strategy.ensure_vector_search_index("documents")
            await self.base_strategy.ensure_vector_search_index("code_examples")
        except Exception as e:
            logger.warning(f"Could not ensure search indexes: {e}")

    def get_setting(self, key: str, default: str = "false") -> str:
        """Get a setting from the credential service or fall back to environment variable."""
        try:
            from ..mongodb_credential_service import mongodb_credential_service as credential_service

            if hasattr(credential_service, "_cache") and credential_service._cache_initialized:
                cached_value = credential_service._cache.get(key)
                if isinstance(cached_value, dict) and cached_value.get("is_encrypted"):
                    encrypted_value = cached_value.get("encrypted_value")
                    if encrypted_value:
                        try:
                            return credential_service._decrypt_value(encrypted_value)
                        except Exception:
                            pass
                elif cached_value:
                    return str(cached_value)
            # Fallback to environment variable
            return os.getenv(key, default)
        except Exception:
            return os.getenv(key, default)

    def get_bool_setting(self, key: str, default: bool = False) -> bool:
        """Get a boolean setting from credential service."""
        value = self.get_setting(key, "false" if not default else "true")
        return value.lower() in ("true", "1", "yes", "on")

    async def search_documents(
        self,
        query: str,
        match_count: int = 5,
        filter_metadata: dict | None = None,
        use_hybrid_search: bool = False,
        cached_api_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Document search with hybrid search capability using MongoDB.

        Args:
            query: Search query string
            match_count: Number of results to return
            filter_metadata: Optional metadata filter dict
            use_hybrid_search: Whether to use hybrid search
            cached_api_key: Deprecated parameter for compatibility

        Returns:
            List of matching documents
        """
        with safe_span(
            "mongodb_rag_search_documents",
            query_length=len(query),
            match_count=match_count,
            hybrid_enabled=use_hybrid_search,
        ) as span:
            try:
                # Create embedding for the query
                query_embedding = await create_embedding(query)

                if not query_embedding:
                    logger.error("Failed to create embedding for query")
                    return []

                if use_hybrid_search:
                    # Use hybrid strategy
                    results = await self.hybrid_strategy.search_documents_hybrid(
                        query=query,
                        query_embedding=query_embedding,
                        match_count=match_count,
                        filter_metadata=filter_metadata,
                    )
                    span.set_attribute("search_mode", "hybrid")
                else:
                    # Use basic vector search from base strategy
                    results = await self.base_strategy.vector_search(
                        query_embedding=query_embedding,
                        match_count=match_count,
                        filter_metadata=filter_metadata,
                        collection_name="documents",
                    )
                    span.set_attribute("search_mode", "vector")

                span.set_attribute("results_found", len(results))
                return results

            except Exception as e:
                logger.error(f"Document search failed: {e}")
                span.set_attribute("error", str(e))
                return []

    async def search_code_examples(
        self,
        query: str,
        match_count: int = 5,
        source_id: str | None = None,
        language: str | None = None,
        cached_api_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search code examples using MongoDB vector search.

        Args:
            query: Search query string
            match_count: Number of results to return
            source_id: Optional source filter
            language: Optional language filter
            cached_api_key: Deprecated parameter for compatibility

        Returns:
            List of matching code examples
        """
        with safe_span(
            "mongodb_rag_search_code_examples",
            query_length=len(query),
            match_count=match_count,
            source_id=source_id,
            language=language,
        ) as span:
            try:
                # Create embedding for the query
                query_embedding = await create_embedding(query)

                if not query_embedding:
                    logger.error("Failed to create embedding for query")
                    return []

                # Build filter metadata
                filter_metadata = {}
                if source_id:
                    filter_metadata["source"] = source_id
                if language:
                    filter_metadata["language"] = language

                # Search code examples
                results = await self.base_strategy.search_code_examples(
                    query_embedding=query_embedding,
                    match_count=match_count,
                    filter_metadata=filter_metadata,
                )

                span.set_attribute("results_found", len(results))
                return results

            except Exception as e:
                logger.error(f"Code examples search failed: {e}")
                span.set_attribute("error", str(e))
                return []

    async def perform_rag_query(
        self,
        query: str,
        match_count: int = 5,
        filter_metadata: dict | None = None,
        cached_api_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Perform a comprehensive RAG query using MongoDB backend.

        This method combines document search and code example search,
        applying all enabled RAG strategies.

        Args:
            query: Search query string
            match_count: Number of results to return per category
            filter_metadata: Optional metadata filters
            cached_api_key: Deprecated parameter for compatibility

        Returns:
            Dictionary containing search results and metadata
        """
        with safe_span(
            "mongodb_rag_perform_query",
            query_length=len(query),
            match_count=match_count,
        ) as span:
            try:
                # Check if hybrid search is enabled
                use_hybrid_search = self.get_bool_setting("USE_HYBRID_SEARCH", True)

                # Search documents
                documents = await self.search_documents(
                    query=query,
                    match_count=match_count,
                    filter_metadata=filter_metadata,
                    use_hybrid_search=use_hybrid_search,
                )

                # Search code examples
                code_examples = await self.search_code_examples(
                    query=query,
                    match_count=match_count,
                    source_id=filter_metadata.get("source") if filter_metadata else None,
                )

                # Combine results
                result = {
                    "query": query,
                    "documents": documents,
                    "code_examples": code_examples,
                    "total_results": len(documents) + len(code_examples),
                    "search_metadata": {
                        "hybrid_search_enabled": use_hybrid_search,
                        "document_count": len(documents),
                        "code_example_count": len(code_examples),
                        "filter_applied": filter_metadata is not None,
                    },
                }

                span.set_attribute("total_results", result["total_results"])
                span.set_attribute("document_results", len(documents))
                span.set_attribute("code_example_results", len(code_examples))

                return result

            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                span.set_attribute("error", str(e))
                return {
                    "query": query,
                    "documents": [],
                    "code_examples": [],
                    "total_results": 0,
                    "error": str(e),
                }

    async def get_available_sources(self) -> list[dict[str, Any]]:
        """
        Get list of available sources from MongoDB.

        Returns:
            List of source information
        """
        try:
            # Get unique sources from documents collection
            pipeline = [
                {"$group": {
                    "_id": "$source_id",
                    "count": {"$sum": 1},
                    "latest_update": {"$max": "$updated_at"}
                }},
                {"$sort": {"latest_update": -1}}
            ]

            sources_cursor = self.db.documents.aggregate(pipeline)
            sources = await sources_cursor.to_list(length=None)

            # Get additional source information from sources collection
            sources_info = []
            for source in sources:
                source_id = source["_id"]

                # Get detailed source info
                source_doc = await self.db.sources.find_one({"source_id": source_id})

                source_info = {
                    "source_id": source_id,
                    "document_count": source["count"],
                    "last_updated": source["latest_update"],
                }

                if source_doc:
                    source_info.update({
                        "title": source_doc.get("title", source_id),
                        "summary": source_doc.get("summary", ""),
                        "knowledge_type": source_doc.get("knowledge_type", "documentation"),
                        "source_type": source_doc.get("source_type", "unknown"),
                        "tags": source_doc.get("tags", []),
                        "word_count": source_doc.get("word_count", 0),
                    })

                sources_info.append(source_info)

            return sources_info

        except Exception as e:
            logger.error(f"Error getting available sources: {e}")
            return []
