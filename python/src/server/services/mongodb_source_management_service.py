"""
MongoDB Source Management Service

Handles source metadata, summaries, and management using MongoDB.
Consolidates both utility functions and class-based service.
"""

import os
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ..config.logfire_config import get_logger, search_logger
from .client_manager import get_mongodb_client

logger = get_logger(__name__)


def _get_model_choice() -> str:
    """Get MODEL_CHOICE with direct fallback."""
    try:
        # Direct cache/env fallback
        from .credential_service import credential_service

        if credential_service._cache_initialized and "MODEL_CHOICE" in credential_service._cache:
            model = credential_service._cache["MODEL_CHOICE"]
        else:
            model = os.getenv("MODEL_CHOICE", "gpt-4.1-nano")
        logger.debug(f"Using model choice: {model}")
        return model
    except Exception as e:
        logger.warning(f"Error getting model choice: {e}, using default")
        return "gpt-4.1-nano"


def extract_source_summary(
    source_id: str, content: str, max_length: int = 500, provider: str = None
) -> str:
    """
    Extract a summary for a source from its content using an LLM.

    This function uses the configured provider to generate a concise summary of the source content.

    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        provider: Optional provider override

    Returns:
        A summary string
    """
    try:
        # Use LLM to generate summary
        from .llm_provider_service import get_llm_client

        llm_client = get_llm_client(provider=provider or _get_model_choice())

        # Create a prompt for summary generation
        prompt = f"""
        Please create a concise summary of this content from {source_id}.
        Focus on the key topics, purpose, and main information covered.
        Keep the summary under {max_length} characters.
        
        Content:
        {content[:2000]}...
        
        Summary:
        """

        try:
            response = llm_client.complete(prompt, max_tokens=100)
            summary = response.text.strip()
            
            # Ensure summary is within max_length
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
                
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate LLM summary for {source_id}: {e}")
            # Fallback to simple truncation
            return content[:max_length-3] + "..." if len(content) > max_length else content

    except Exception as e:
        logger.error(f"Error extracting summary for {source_id}: {e}")
        # Return first paragraph or truncated content as fallback
        first_paragraph = content.split('\n\n')[0] if content else ""
        if len(first_paragraph) > max_length:
            return first_paragraph[:max_length-3] + "..."
        return first_paragraph or f"Content from {source_id}"


def generate_source_title_and_metadata(content: str, source_id: str, provider: str = None) -> dict[str, Any]:
    """
    Generate title and metadata for a source using content analysis.

    Args:
        content: The content to analyze
        source_id: The source identifier
        provider: Optional provider override

    Returns:
        Dictionary with title and metadata
    """
    try:
        # Extract title from content (first meaningful line)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Look for title patterns
        title = source_id  # fallback
        
        for line in lines[:10]:  # Check first 10 lines
            # Look for markdown headers
            if line.startswith('# '):
                title = line[2:].strip()
                break
            # Look for HTML titles
            elif '<title>' in line.lower():
                import re
                match = re.search(r'<title[^>]*>(.*?)</title>', line, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    break
            # Look for other header patterns
            elif len(line) > 10 and len(line) < 200 and not line.startswith('http'):
                title = line
                break

        # Generate metadata
        word_count = len(content.split()) if content else 0
        char_count = len(content) if content else 0
        
        # Extract basic content stats
        has_code = '```' in content or '<code>' in content.lower()
        has_links = 'http' in content.lower()
        
        metadata = {
            'title': title[:200],  # Limit title length
            'word_count': word_count,
            'char_count': char_count,
            'has_code_blocks': has_code,
            'has_links': has_links,
            'content_type': 'documentation' if has_code else 'general',
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error generating title/metadata for {source_id}: {e}")
        return {
            'title': source_id,
            'word_count': 0,
            'char_count': 0,
            'has_code_blocks': False,
            'has_links': False,
            'content_type': 'general',
        }


async def update_source_info(
    db: AsyncIOMotorDatabase,
    source_id: str,
    summary: str,
    word_count: int,
    preview_content: str = "",
    knowledge_type: str = "documentation",
    tags: list[str] = None,
) -> bool:
    """
    Update source information in MongoDB.

    Args:
        db: MongoDB database instance
        source_id: Source identifier
        summary: Source summary
        word_count: Total word count
        preview_content: Preview content for title generation
        knowledge_type: Type of knowledge
        tags: Optional tags

    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate title and metadata from content
        title_meta = generate_source_title_and_metadata(preview_content, source_id)
        
        current_time = datetime.utcnow()
        
        # Create source document
        source_doc = {
            'source_id': source_id,
            'title': title_meta['title'],
            'summary': summary,
            'word_count': word_count,
            'knowledge_type': knowledge_type,
            'source_type': 'web',  # Default to web, can be overridden
            'tags': tags or [],
            'metadata': {
                'char_count': title_meta.get('char_count', 0),
                'has_code_blocks': title_meta.get('has_code_blocks', False),
                'has_links': title_meta.get('has_links', False),
                'content_type': title_meta.get('content_type', 'documentation'),
            },
            'updated_at': current_time,
        }
        
        # Upsert source document
        result = await db.sources.update_one(
            {'source_id': source_id},
            {
                '$set': source_doc,
                '$setOnInsert': {'created_at': current_time}
            },
            upsert=True
        )
        
        logger.info(f"Updated source info for {source_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update source info for {source_id}: {e}")
        return False


class MongoDBSourceManagementService:
    """Service for managing source metadata and operations using MongoDB."""

    def __init__(self, db: AsyncIOMotorDatabase = None):
        """Initialize with MongoDB database."""
        self.db = db or get_mongodb_client()

    async def get_source(self, source_id: str) -> dict[str, Any] | None:
        """Get source information by ID."""
        try:
            source = await self.db.sources.find_one({'source_id': source_id})
            if source:
                # Convert ObjectId to string
                source['_id'] = str(source['_id'])
                return source
            return None
        except Exception as e:
            logger.error(f"Error getting source {source_id}: {e}")
            return None

    async def list_sources(self, knowledge_type: str = None, limit: int = 100) -> list[dict[str, Any]]:
        """List all sources with optional filtering."""
        try:
            query = {}
            if knowledge_type:
                query['knowledge_type'] = knowledge_type
            
            cursor = self.db.sources.find(query).sort('updated_at', -1).limit(limit)
            sources = await cursor.to_list(length=None)
            
            # Convert ObjectIds to strings
            for source in sources:
                source['_id'] = str(source['_id'])
            
            return sources
            
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []

    def delete_source(self, source_id: str) -> tuple[bool, dict[str, Any]]:
        """
        Delete a source and all associated data.
        
        Returns:
            Tuple of (success, result_data)
        """
        try:
            import asyncio
            
            # Run the async delete operation
            result = asyncio.create_task(self._delete_source_async(source_id))
            loop = asyncio.get_event_loop()
            
            if loop.is_running():
                # If we're already in an async context, we need to handle this differently
                # For now, return a placeholder result
                return True, {"message": f"Source {source_id} deletion initiated"}
            else:
                return loop.run_until_complete(result)
                
        except Exception as e:
            logger.error(f"Error deleting source {source_id}: {e}")
            return False, {"error": str(e)}

    async def _delete_source_async(self, source_id: str) -> tuple[bool, dict[str, Any]]:
        """Async implementation of source deletion."""
        try:
            # Delete documents
            doc_result = await self.db.documents.delete_many({'source_id': source_id})
            
            # Delete code examples
            code_result = await self.db.code_examples.delete_many({'source_id': source_id})
            
            # Delete source metadata
            source_result = await self.db.sources.delete_one({'source_id': source_id})
            
            deleted_counts = {
                'documents': doc_result.deleted_count,
                'code_examples': code_result.deleted_count,
                'source_metadata': source_result.deleted_count,
            }
            
            logger.info(f"Deleted source {source_id}: {deleted_counts}")
            
            return True, {
                "message": f"Successfully deleted source {source_id}",
                "deleted_counts": deleted_counts
            }
            
        except Exception as e:
            logger.error(f"Error in async delete for {source_id}: {e}")
            return False, {"error": str(e)}

    async def update_source_metadata(
        self,
        source_id: str,
        title: str = None,
        summary: str = None,
        knowledge_type: str = None,
        tags: list[str] = None,
    ) -> bool:
        """Update source metadata."""
        try:
            updates = {}
            if title:
                updates['title'] = title
            if summary:
                updates['summary'] = summary
            if knowledge_type:
                updates['knowledge_type'] = knowledge_type
            if tags is not None:
                updates['tags'] = tags
            
            if not updates:
                return True
                
            updates['updated_at'] = datetime.utcnow()
            
            result = await self.db.sources.update_one(
                {'source_id': source_id},
                {'$set': updates}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating source metadata for {source_id}: {e}")
            return False


# For backward compatibility, alias the MongoDB version
SourceManagementService = MongoDBSourceManagementService


# Update the utility function to work with MongoDB
async def update_source_info_compat(
    client_or_db,  # Can be either MongoDB db or old Supabase client for compatibility
    source_id: str,
    summary: str,
    word_count: int,
    preview_content: str = "",
    knowledge_type: str = "documentation",
    tags: list[str] = None,
) -> bool:
    """Compatibility wrapper for update_source_info."""
    # Check if it's a MongoDB database or Supabase client
    if hasattr(client_or_db, 'sources'):
        # It's a MongoDB database
        return await update_source_info(
            client_or_db, source_id, summary, word_count, 
            preview_content, knowledge_type, tags
        )
    else:
        # For backward compatibility, get MongoDB client
        db = await get_mongodb_client()
        return await update_source_info(
            db, source_id, summary, word_count,
            preview_content, knowledge_type, tags
        )