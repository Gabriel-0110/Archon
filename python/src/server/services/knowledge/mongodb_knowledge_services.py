"""
MongoDB Knowledge Services

Handles all knowledge item CRUD operations and data transformations using MongoDB.
"""

from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...config.logfire_config import safe_logfire_error
from ...config.mongodb_config import get_mongodb_database


class MongoDBKnowledgeItemService:
    """
    Service for managing knowledge items including listing, filtering, updating, and deletion.
    """

    def __init__(self, db: AsyncIOMotorDatabase = None):
        """
        Initialize the knowledge item service.

        Args:
            db: The MongoDB database instance
        """
        self.db = db or get_mongodb_database()

    async def list_items(
        self,
        page: int = 1,
        per_page: int = 20,
        knowledge_type: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        """
        List knowledge items with filtering and pagination.

        Args:
            page: Page number (1-based)
            per_page: Items per page
            knowledge_type: Optional filter by knowledge type
            search: Optional search term

        Returns:
            Dictionary with items, pagination info, and totals
        """
        try:
            # Build query
            query = {}
            if knowledge_type:
                query["knowledge_type"] = knowledge_type
            if search:
                query["$text"] = {"$search": search}

            # Calculate skip and limit
            skip = (page - 1) * per_page

            # Get total count
            total_count = await self.db.sources.count_documents(query)

            # Get items with pagination
            cursor = self.db.sources.find(query).sort("updated_at", -1).skip(skip).limit(per_page)
            sources = await cursor.to_list(length=None)

            # Convert ObjectId to string and format response
            items = []
            for source in sources:
                item = {
                    "source_id": source.get("source_id"),
                    "title": source.get("title", source.get("source_id")),
                    "knowledge_type": source.get("knowledge_type", "documentation"),
                    "source_type": source.get("source_type", "web"),
                    "word_count": source.get("word_count", 0),
                    "created_at": source.get("created_at"),
                    "updated_at": source.get("updated_at"),
                    "tags": source.get("tags", []),
                    "summary": source.get("summary", ""),
                }
                items.append(item)

            # Calculate pagination info
            total_pages = (total_count + per_page - 1) // per_page
            has_next = page < total_pages
            has_prev = page > 1

            return {
                "items": items,
                "pagination": {
                    "current_page": page,
                    "per_page": per_page,
                    "total_pages": total_pages,
                    "total_count": total_count,
                    "has_next": has_next,
                    "has_prev": has_prev,
                },
                "totals": {
                    "total_sources": total_count,
                },
            }

        except Exception as e:
            safe_logfire_error(f"Error listing knowledge items: {e}")
            return {
                "items": [],
                "pagination": {
                    "current_page": page,
                    "per_page": per_page,
                    "total_pages": 0,
                    "total_count": 0,
                    "has_next": False,
                    "has_prev": False,
                },
                "totals": {
                    "total_sources": 0,
                },
            }

    async def get_item(self, source_id: str) -> dict[str, Any] | None:
        """Get a specific knowledge item by source ID."""
        try:
            source = await self.db.sources.find_one({"source_id": source_id})
            if not source:
                return None

            return {
                "source_id": source.get("source_id"),
                "title": source.get("title", source.get("source_id")),
                "knowledge_type": source.get("knowledge_type", "documentation"),
                "source_type": source.get("source_type", "web"),
                "word_count": source.get("word_count", 0),
                "created_at": source.get("created_at"),
                "updated_at": source.get("updated_at"),
                "tags": source.get("tags", []),
                "summary": source.get("summary", ""),
                "metadata": source.get("metadata", {}),
            }

        except Exception as e:
            safe_logfire_error(f"Error getting knowledge item {source_id}: {e}")
            return None

    async def update_item(self, source_id: str, updates: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Update a knowledge item."""
        try:
            # Add update timestamp
            updates["updated_at"] = datetime.utcnow()

            result = await self.db.sources.update_one(
                {"source_id": source_id},
                {"$set": updates}
            )

            if result.modified_count > 0:
                updated_item = await self.get_item(source_id)
                return True, updated_item or {}
            else:
                return False, {"error": "Item not found or no changes made"}

        except Exception as e:
            safe_logfire_error(f"Error updating knowledge item {source_id}: {e}")
            return False, {"error": str(e)}

    async def get_available_sources(self) -> list[dict[str, Any]]:
        """Get list of available sources for RAG queries."""
        try:
            # Get unique sources with their metadata
            pipeline = [
                {"$group": {
                    "_id": "$source_id",
                    "title": {"$first": "$title"},
                    "knowledge_type": {"$first": "$knowledge_type"},
                    "source_type": {"$first": "$source_type"},
                    "word_count": {"$first": "$word_count"},
                    "updated_at": {"$first": "$updated_at"},
                    "tags": {"$first": "$tags"},
                }},
                {"$sort": {"updated_at": -1}}
            ]

            cursor = self.db.sources.aggregate(pipeline)
            sources = await cursor.to_list(length=None)

            result = []
            for source in sources:
                result.append({
                    "source_id": source["_id"],
                    "title": source.get("title", source["_id"]),
                    "knowledge_type": source.get("knowledge_type", "documentation"),
                    "source_type": source.get("source_type", "web"),
                    "word_count": source.get("word_count", 0),
                    "last_updated": source.get("updated_at"),
                    "tags": source.get("tags", []),
                })

            return result

        except Exception as e:
            safe_logfire_error(f"Error getting available sources: {e}")
            return []


class MongoDBDatabaseMetricsService:
    """
    Service for getting database metrics and statistics using MongoDB.
    """

    def __init__(self, db: AsyncIOMotorDatabase = None):
        """
        Initialize the database metrics service.

        Args:
            db: The MongoDB database instance
        """
        self.db = db or get_mongodb_database()

    async def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive database metrics."""
        try:
            # Get collection stats
            sources_count = await self.db.sources.count_documents({})
            documents_count = await self.db.documents.count_documents({})
            code_examples_count = await self.db.code_examples.count_documents({})
            projects_count = await self.db.projects.count_documents({})
            tasks_count = await self.db.tasks.count_documents({})

            # Get knowledge type breakdown
            pipeline = [
                {"$group": {
                    "_id": "$knowledge_type",
                    "count": {"$sum": 1}
                }}
            ]
            knowledge_types_cursor = self.db.sources.aggregate(pipeline)
            knowledge_types = await knowledge_types_cursor.to_list(length=None)

            knowledge_type_breakdown = {}
            for kt in knowledge_types:
                knowledge_type_breakdown[kt["_id"] or "unknown"] = kt["count"]

            # Get total word count
            pipeline = [
                {"$group": {
                    "_id": None,
                    "total_words": {"$sum": "$word_count"}
                }}
            ]
            word_count_cursor = self.db.sources.aggregate(pipeline)
            word_count_result = await word_count_cursor.to_list(length=1)
            total_words = word_count_result[0]["total_words"] if word_count_result else 0

            return {
                "collections": {
                    "sources": sources_count,
                    "documents": documents_count,
                    "code_examples": code_examples_count,
                    "projects": projects_count,
                    "tasks": tasks_count,
                },
                "knowledge_types": knowledge_type_breakdown,
                "content_stats": {
                    "total_words": total_words,
                    "average_words_per_source": total_words // max(sources_count, 1),
                },
                "database_type": "MongoDB",
            }

        except Exception as e:
            safe_logfire_error(f"Error getting database metrics: {e}")
            return {
                "collections": {},
                "knowledge_types": {},
                "content_stats": {},
                "error": str(e),
            }


# For backward compatibility
KnowledgeItemService = MongoDBKnowledgeItemService
DatabaseMetricsService = MongoDBDatabaseMetricsService
