"""
Storage Services

Handles document and code storage operations using MongoDB.
"""

from .base_storage_service import BaseStorageService
from .mongodb_code_storage import (
    add_code_examples_to_mongodb,
    get_code_examples_from_mongodb,
    delete_code_examples_by_source,
    generate_code_summaries_batch,
)
from .mongodb_document_storage import add_documents_to_mongodb
from .mongodb_storage_services import MongoDBDocumentStorageService

# For backward compatibility, alias the MongoDB versions
DocumentStorageService = MongoDBDocumentStorageService
add_documents_to_supabase = add_documents_to_mongodb
add_code_examples_to_supabase = add_code_examples_to_mongodb

__all__ = [
    # Base service
    "BaseStorageService",
    # Service classes
    "DocumentStorageService", 
    "MongoDBDocumentStorageService",
    # Document storage utilities
    "add_documents_to_supabase",  # Backward compatibility alias
    "add_documents_to_mongodb",
    # Code storage utilities
    "add_code_examples_to_supabase",  # Backward compatibility alias
    "add_code_examples_to_mongodb",
    "get_code_examples_from_mongodb",
    "delete_code_examples_by_source",
    "generate_code_summaries_batch",
]
