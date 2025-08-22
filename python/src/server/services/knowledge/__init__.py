"""
Knowledge Services Package - Now using MongoDB

Contains services for knowledge management operations.
"""
from .mongodb_knowledge_services import (
    MongoDBKnowledgeItemService,
    MongoDBDatabaseMetricsService,
)

# For backward compatibility
KnowledgeItemService = MongoDBKnowledgeItemService
DatabaseMetricsService = MongoDBDatabaseMetricsService

__all__ = [
    'KnowledgeItemService',
    'DatabaseMetricsService',
    'MongoDBKnowledgeItemService',
    'MongoDBDatabaseMetricsService',
]
