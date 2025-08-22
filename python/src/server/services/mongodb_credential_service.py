"""
MongoDB Credential management service for Archon backend

Handles loading, storing, and accessing credentials with encryption for sensitive values.
Credentials include API keys, service credentials, and application configuration.
"""

import base64
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..config.logfire_config import get_logger
from ..config.mongodb_config import get_mongodb_database

logger = get_logger(__name__)


@dataclass
class CredentialItem:
    """Represents a credential/setting item."""

    key: str
    value: str | None = None
    encrypted_value: str | None = None
    is_encrypted: bool = False
    category: str | None = None
    description: str | None = None


class MongoDBCredentialService:
    """Service for managing application credentials and configuration using MongoDB."""

    def __init__(self):
        self._db: AsyncIOMotorDatabase | None = None
        self._cache: dict[str, Any] = {}
        self._cache_initialized = False
        self._rag_settings_cache: dict[str, Any] | None = None
        self._rag_cache_timestamp: float | None = None
        self._rag_cache_ttl = 300  # 5 minutes TTL for RAG settings cache

    def _get_mongodb_db(self) -> AsyncIOMotorDatabase:
        """Get MongoDB database instance."""
        if self._db is None:
            self._db = get_mongodb_database()
        return self._db

    def _generate_encryption_key(self, password: str) -> bytes:
        """Generate encryption key from password."""
        salt = b"archon_salt_2023"  # Use a fixed salt for consistency
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def _encrypt_value(self, value: str) -> str:
        """Encrypt a value using application password."""
        # Use a default password if not set (for development)
        password = os.getenv("ARCHON_ENCRYPTION_PASSWORD", "default_dev_password_2023")
        key = self._generate_encryption_key(password)
        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a value using application password."""
        try:
            password = os.getenv("ARCHON_ENCRYPTION_PASSWORD", "default_dev_password_2023")
            key = self._generate_encryption_key(password)
            f = Fernet(key)
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = f.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            raise ValueError("Failed to decrypt credential value") from e

    async def _ensure_credentials_collection(self):
        """Ensure credentials collection exists with proper indexes."""
        db = self._get_mongodb_db()

        # Create unique index on key
        await db.credentials.create_index("key", unique=True)
        await db.credentials.create_index("category")

    async def load_cache(self):
        """Load all credentials into cache."""
        try:
            await self._ensure_credentials_collection()
            db = self._get_mongodb_db()

            cursor = db.credentials.find()
            credentials = await cursor.to_list(length=None)

            self._cache = {}
            for cred in credentials:
                key = cred["key"]

                if cred.get("is_encrypted", False):
                    # Store encrypted credentials with metadata
                    self._cache[key] = {
                        "encrypted_value": cred["encrypted_value"],
                        "is_encrypted": True,
                        "category": cred.get("category"),
                    }
                else:
                    # Store plain text credentials directly
                    self._cache[key] = cred["value"]

            self._cache_initialized = True
            logger.info(f"Loaded {len(self._cache)} credentials into cache")

        except Exception as e:
            logger.error(f"Failed to load credentials cache: {e}")
            self._cache = {}
            self._cache_initialized = True

    async def get_credential(self, key: str, default: str | None = None, decrypt: bool = True) -> str | None:
        """Get a credential value with automatic decryption."""
        if not self._cache_initialized:
            await self.load_cache()

        # Check cache first
        if key in self._cache:
            cached_value = self._cache[key]

            if isinstance(cached_value, dict) and cached_value.get("is_encrypted"):
                if decrypt:
                    try:
                        encrypted_value = cached_value.get("encrypted_value")
                        if encrypted_value:
                            return self._decrypt_value(encrypted_value)
                    except Exception as e:
                        logger.error(f"Failed to decrypt credential {key}: {e}")
                        return default
                else:
                    return cached_value.get("encrypted_value")
            else:
                return str(cached_value) if cached_value is not None else default

        # Not in cache, check environment variables as fallback
        env_value = os.getenv(key)
        if env_value:
            return env_value

        return default

    async def set_credential(self, key: str, value: str, encrypt: bool = True, category: str | None = None, description: str | None = None) -> bool:
        """Store a credential with optional encryption."""
        try:
            await self._ensure_credentials_collection()
            db = self._get_mongodb_db()

            current_time = datetime.utcnow()

            if encrypt:
                encrypted_value = self._encrypt_value(value)
                credential_doc = {
                    "key": key,
                    "value": None,  # Don't store plain text when encrypted
                    "encrypted_value": encrypted_value,
                    "is_encrypted": True,
                    "category": category,
                    "description": description,
                    "updated_at": current_time,
                }

                # Update cache
                self._cache[key] = {
                    "encrypted_value": encrypted_value,
                    "is_encrypted": True,
                    "category": category,
                }
            else:
                credential_doc = {
                    "key": key,
                    "value": value,
                    "encrypted_value": None,
                    "is_encrypted": False,
                    "category": category,
                    "description": description,
                    "updated_at": current_time,
                }

                # Update cache
                self._cache[key] = value

            # Upsert credential
            await db.credentials.update_one(
                {"key": key},
                {
                    "$set": credential_doc,
                    "$setOnInsert": {"created_at": current_time}
                },
                upsert=True
            )

            logger.info(f"Successfully stored credential: {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to store credential {key}: {e}")
            return False

    async def delete_credential(self, key: str) -> bool:
        """Delete a credential."""
        try:
            db = self._get_mongodb_db()

            result = await db.credentials.delete_one({"key": key})

            # Remove from cache
            if key in self._cache:
                del self._cache[key]

            logger.info(f"Deleted credential: {key}")
            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to delete credential {key}: {e}")
            return False

    async def get_credentials_by_category(self, category: str) -> dict[str, Any]:
        """Get all credentials in a specific category."""
        if not self._cache_initialized:
            await self.load_cache()

        # Check if this is RAG settings and use cache
        if category == "rag_strategy":
            current_time = time.time()
            if (self._rag_settings_cache is not None and
                self._rag_cache_timestamp is not None and
                current_time - self._rag_cache_timestamp < self._rag_cache_ttl):
                return self._rag_settings_cache

        try:
            db = self._get_mongodb_db()
            cursor = db.credentials.find({"category": category})
            credentials = await cursor.to_list(length=None)

            result = {}
            for cred in credentials:
                key = cred["key"]

                if cred.get("is_encrypted", False):
                    try:
                        encrypted_value = cred["encrypted_value"]
                        if encrypted_value:
                            result[key] = self._decrypt_value(encrypted_value)
                    except Exception as e:
                        logger.error(f"Failed to decrypt credential {key}: {e}")
                        continue
                else:
                    result[key] = cred["value"]

            # Cache RAG settings
            if category == "rag_strategy":
                self._rag_settings_cache = result
                self._rag_cache_timestamp = time.time()

            return result

        except Exception as e:
            logger.error(f"Failed to get credentials for category {category}: {e}")
            return {}

    async def list_credentials(self, include_values: bool = False) -> list[CredentialItem]:
        """List all credentials."""
        try:
            db = self._get_mongodb_db()
            cursor = db.credentials.find()
            credentials = await cursor.to_list(length=None)

            result = []
            for cred in credentials:
                item = CredentialItem(
                    key=cred["key"],
                    is_encrypted=cred.get("is_encrypted", False),
                    category=cred.get("category"),
                    description=cred.get("description"),
                )

                if include_values:
                    if cred.get("is_encrypted", False):
                        try:
                            encrypted_value = cred["encrypted_value"]
                            if encrypted_value:
                                item.value = self._decrypt_value(encrypted_value)
                        except Exception as e:
                            logger.error(f"Failed to decrypt credential {cred['key']}: {e}")
                            item.value = "[DECRYPTION_FAILED]"
                    else:
                        item.value = cred["value"]

                result.append(item)

            return result

        except Exception as e:
            logger.error(f"Failed to list credentials: {e}")
            return []

    def _validate_openai_key(self, api_key: str) -> tuple[bool, str]:
        """Validate OpenAI API key format."""
        if not api_key or not api_key.strip():
            return False, "API key cannot be empty"

        api_key = api_key.strip()

        if not api_key.startswith("sk-"):
            return False, "OpenAI API key must start with 'sk-'"

        if len(api_key) < 20:
            return False, "API key appears to be too short"

        # Check for common placeholder values
        placeholder_patterns = [
            r"^sk-[x]+$",
            r"^sk-your[_-]?key[_-]?here$",
            r"^sk-insert[_-]?key[_-]?here$",
            r"^sk-replace[_-]?with[_-]?your[_-]?key$",
        ]

        for pattern in placeholder_patterns:
            if re.match(pattern, api_key, re.IGNORECASE):
                return False, "Please replace the placeholder with your actual API key"

        return True, "Valid format"


# Global instance
mongodb_credential_service = MongoDBCredentialService()

# For backward compatibility
credential_service = mongodb_credential_service


async def initialize_credentials():
    """Initialize/reload credentials cache."""
    await credential_service.load_cache()
