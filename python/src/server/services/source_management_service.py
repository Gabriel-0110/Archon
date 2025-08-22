"""
Source Management Service

This module now uses MongoDB instead of Supabase.
Importing the MongoDB implementation for backward compatibility.
"""

# Import MongoDB implementations for backward compatibility
from .mongodb_source_management_service import (
    MongoDBSourceManagementService,
    extract_source_summary,
    generate_source_title_and_metadata,
    update_source_info_compat as update_source_info,
)

# For backward compatibility, alias the MongoDB version
SourceManagementService = MongoDBSourceManagementService

__all__ = [
    "SourceManagementService",
    "MongoDBSourceManagementService", 
    "extract_source_summary",
    "generate_source_title_and_metadata",
    "update_source_info",
]