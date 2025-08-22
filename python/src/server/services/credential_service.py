"""
Credential management service for Archon backend

This module now uses MongoDB instead of Supabase.
Importing the MongoDB implementation for backward compatibility.
"""

# Import MongoDB implementations for backward compatibility
from .mongodb_credential_service import (
    CredentialItem,
    MongoDBCredentialService,
    initialize_credentials,
    mongodb_credential_service,
)

# For backward compatibility, alias the MongoDB version
CredentialService = MongoDBCredentialService
credential_service = mongodb_credential_service

__all__ = [
    "CredentialItem",
    "CredentialService",
    "MongoDBCredentialService",
    "credential_service",
    "mongodb_credential_service",
    "initialize_credentials",
]
