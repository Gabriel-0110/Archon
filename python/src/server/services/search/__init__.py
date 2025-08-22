"""
Search Services

Consolidated search and RAG functionality using MongoDB.
"""

# MongoDB implementations  
from .mongodb_rag_service import MongoDBRAGService
from .mongodb_search_strategies import MongoDBBaseSearchStrategy, MongoDBHybridSearchStrategy

# For backward compatibility, alias the MongoDB versions
RAGService = MongoDBRAGService
BaseSearchStrategy = MongoDBBaseSearchStrategy
HybridSearchStrategy = MongoDBHybridSearchStrategy

__all__ = [
    # Main service classes
    "RAGService",  # Backward compatibility alias
    "MongoDBRAGService",
    # Strategy classes
    "BaseSearchStrategy",  # Backward compatibility alias
    "HybridSearchStrategy",  # Backward compatibility alias
    "MongoDBBaseSearchStrategy",
    "MongoDBHybridSearchStrategy",
]
