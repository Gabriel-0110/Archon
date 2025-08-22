"""
MongoDB Code Storage Service

Handles extraction and storage of code examples from documents using MongoDB.
"""

import asyncio
import os
import re
from collections.abc import Callable
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from ...config.logfire_config import search_logger


def _get_model_choice() -> str:
    """Get MODEL_CHOICE with direct fallback."""
    try:
        # Direct cache/env fallback
        from ..credential_service import credential_service

        if credential_service._cache_initialized and "MODEL_CHOICE" in credential_service._cache:
            model = credential_service._cache["MODEL_CHOICE"]
        else:
            model = os.getenv("MODEL_CHOICE", "gpt-4.1-nano")
        search_logger.debug(f"Using model choice: {model}")
        return model
    except Exception as e:
        search_logger.warning(f"Error getting model choice: {e}, using default")
        return "gpt-4.1-nano"


def _get_max_workers() -> int:
    """Get max workers from environment, defaulting to 3."""
    return int(os.getenv("CONTEXTUAL_EMBEDDINGS_MAX_WORKERS", "3"))


def _normalize_code_for_comparison(code: str) -> str:
    """
    Normalize code for similarity comparison by removing version-specific variations.

    Args:
        code: The code string to normalize

    Returns:
        Normalized code string for comparison
    """
    # Remove extra whitespace and normalize line endings
    normalized = re.sub(r"\s+", " ", code.strip())

    # Remove common version-specific imports that don't change functionality
    # Handle typing imports variations
    normalized = re.sub(r"from typing_extensions import", "from typing import", normalized)
    normalized = re.sub(r"from typing import Annotated[^,\n]*,?", "", normalized)
    normalized = re.sub(r"from typing_extensions import Annotated[^,\n]*,?", "", normalized)

    # Remove Annotated wrapper variations for comparison
    # This handles: Annotated[type, dependency] -> type
    normalized = re.sub(r"Annotated\[\s*([^,\]]+)[^]]*\]", r"\1", normalized)

    # Normalize common FastAPI parameter patterns
    normalized = re.sub(r":\s*Annotated\[[^\]]+\]\s*=", "=", normalized)

    # Remove trailing commas and normalize punctuation spacing
    normalized = re.sub(r",\s*\)", ")", normalized)
    normalized = re.sub(r",\s*]", "]", normalized)

    return normalized


def _calculate_code_similarity(code1: str, code2: str) -> float:
    """
    Calculate similarity between two code strings using normalized comparison.

    Args:
        code1: First code string
        code2: Second code string

    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    # Normalize both code strings for comparison
    norm1 = _normalize_code_for_comparison(code1)
    norm2 = _normalize_code_for_comparison(code2)

    # Use difflib's SequenceMatcher for similarity calculation
    similarity = SequenceMatcher(None, norm1, norm2).ratio()

    return similarity


def _select_best_code_variant(similar_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Select the best variant from a list of similar code blocks.
    Prioritizes more complete examples and recent content.

    Args:
        similar_blocks: List of similar code blocks

    Returns:
        The best code block variant
    """
    if not similar_blocks:
        return {}

    if len(similar_blocks) == 1:
        return similar_blocks[0]

    # Score each block based on multiple criteria
    def score_block(block: dict[str, Any]) -> float:
        score = 0.0

        # Length score - longer blocks are generally more complete
        code_length = len(block.get("code", ""))
        score += min(code_length / 1000, 1.0) * 0.3

        # Context score - more context is better
        context_length = len(block.get("context", ""))
        score += min(context_length / 500, 1.0) * 0.2

        # Language specificity - prefer blocks with identified language
        if block.get("language") and block.get("language") != "unknown":
            score += 0.2

        # Line range score - prefer blocks with line information
        if block.get("line_start") and block.get("line_end"):
            line_range = block.get("line_end", 0) - block.get("line_start", 0)
            score += min(line_range / 50, 1.0) * 0.1

        # Metadata richness score
        metadata = block.get("metadata", {})
        if isinstance(metadata, dict):
            score += min(len(metadata) / 10, 1.0) * 0.2

        return score

    # Find the block with the highest score
    best_block = max(similar_blocks, key=score_block)

    search_logger.debug(
        f"Selected best variant from {len(similar_blocks)} similar blocks: "
        f"length={len(best_block.get('code', ''))}, "
        f"language={best_block.get('language', 'unknown')}"
    )

    return best_block


async def add_code_examples_to_mongodb(
    db: AsyncIOMotorDatabase,
    code_examples: list[dict[str, Any]],
    batch_size: int = 50,
    similarity_threshold: float = 0.85,
    progress_callback: Callable | None = None,
    cancellation_check: Callable | None = None,
) -> dict[str, Any]:
    """
    Add code examples to MongoDB with deduplication and batch processing.

    Args:
        db: MongoDB database instance
        code_examples: List of code examples to store
        batch_size: Number of examples to process per batch
        similarity_threshold: Threshold for considering code blocks similar
        progress_callback: Optional callback for progress updates
        cancellation_check: Optional function to check for cancellation

    Returns:
        Dictionary with processing results
    """
    if not code_examples:
        return {"stored": 0, "duplicates_removed": 0, "errors": 0}

    search_logger.info(f"Processing {len(code_examples)} code examples")

    # Progress reporting helper
    async def report_progress(message: str, percentage: int):
        if progress_callback and asyncio.iscoroutinefunction(progress_callback):
            await progress_callback(message, percentage)

    stored_count = 0
    duplicates_removed = 0
    error_count = 0
    total_examples = len(code_examples)

    # Process in batches
    for batch_start in range(0, total_examples, batch_size):
        # Check for cancellation
        if cancellation_check:
            cancellation_check()

        batch_end = min(batch_start + batch_size, total_examples)
        batch = code_examples[batch_start:batch_end]

        progress_pct = int((batch_start / total_examples) * 100)
        await report_progress(
            f"Processing code examples batch {batch_start//batch_size + 1}...", progress_pct
        )

        # Deduplicate within batch and against existing data
        deduplicated_batch = []

        for example in batch:
            try:
                code = example.get("code", "").strip()
                if not code:
                    continue

                # Check against existing code examples in database
                similar_existing = await db.code_examples.find({
                    "language": example.get("language"),
                    "source_id": example.get("source_id")
                }).to_list(length=None)

                is_duplicate = False
                for existing in similar_existing:
                    existing_code = existing.get("code", "")
                    similarity = _calculate_code_similarity(code, existing_code)

                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        duplicates_removed += 1
                        search_logger.debug(
                            f"Duplicate code found (similarity: {similarity:.2f}) for "
                            f"{example.get('language', 'unknown')} in {example.get('source_id', 'unknown')}"
                        )
                        break

                if not is_duplicate:
                    # Check against current batch for duplicates
                    batch_duplicate = False
                    for existing_in_batch in deduplicated_batch:
                        batch_similarity = _calculate_code_similarity(
                            code, existing_in_batch.get("code", "")
                        )
                        if batch_similarity >= similarity_threshold:
                            # Keep the better version
                            if _select_best_code_variant([example, existing_in_batch]) == example:
                                # Replace the existing one in batch
                                deduplicated_batch.remove(existing_in_batch)
                                deduplicated_batch.append(example)
                            batch_duplicate = True
                            duplicates_removed += 1
                            break

                    if not batch_duplicate:
                        deduplicated_batch.append(example)

            except Exception as e:
                search_logger.error(f"Error processing code example: {e}")
                error_count += 1

        # Prepare MongoDB documents
        if deduplicated_batch:
            current_time = datetime.utcnow()
            mongo_docs = []

            for example in deduplicated_batch:
                try:
                    # Extract source ID from URL if not provided
                    source_id = example.get("source_id")
                    if not source_id and example.get("url"):
                        parsed_url = urlparse(example["url"])
                        source_id = parsed_url.netloc or parsed_url.path

                    doc = {
                        "_id": ObjectId(),
                        "source_id": source_id,
                        "url": example.get("url"),
                        "language": example.get("language", "unknown"),
                        "code": example.get("code"),
                        "context": example.get("context", ""),
                        "line_start": example.get("line_start"),
                        "line_end": example.get("line_end"),
                        "metadata": example.get("metadata", {}),
                        "created_at": current_time,
                        "updated_at": current_time,
                    }
                    mongo_docs.append(doc)

                except Exception as e:
                    search_logger.error(f"Error preparing code example document: {e}")
                    error_count += 1

            # Insert batch into MongoDB
            if mongo_docs:
                try:
                    result = await db.code_examples.insert_many(mongo_docs)
                    batch_stored = len(result.inserted_ids)
                    stored_count += batch_stored

                    search_logger.info(
                        f"Stored {batch_stored} code examples in batch "
                        f"{batch_start//batch_size + 1}"
                    )

                except Exception as e:
                    search_logger.error(f"Error storing batch to MongoDB: {e}")
                    error_count += len(mongo_docs)

        # Brief pause between batches
        await asyncio.sleep(0.1)

    # Final progress update
    await report_progress("Code examples processing completed", 100)

    result = {
        "stored": stored_count,
        "duplicates_removed": duplicates_removed,
        "errors": error_count,
        "total_processed": total_examples,
    }

    search_logger.info(
        f"Code examples processing completed: {stored_count} stored, "
        f"{duplicates_removed} duplicates removed, {error_count} errors"
    )

    return result


async def get_code_examples_from_mongodb(
    db: AsyncIOMotorDatabase,
    source_id: str | None = None,
    language: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Retrieve code examples from MongoDB with optional filtering.

    Args:
        db: MongoDB database instance
        source_id: Optional source ID filter
        language: Optional language filter
        limit: Maximum number of examples to return

    Returns:
        List of code examples
    """
    query = {}

    if source_id:
        query["source_id"] = source_id
    if language:
        query["language"] = language

    try:
        cursor = db.code_examples.find(query).limit(limit).sort("created_at", -1)
        examples = await cursor.to_list(length=None)

        # Convert ObjectId to string for JSON serialization
        for example in examples:
            example["_id"] = str(example["_id"])

        return examples

    except Exception as e:
        search_logger.error(f"Error retrieving code examples: {e}")
        return []


async def delete_code_examples_by_source(
    db: AsyncIOMotorDatabase,
    source_id: str,
) -> int:
    """
    Delete all code examples for a specific source.

    Args:
        db: MongoDB database instance
        source_id: Source ID to delete examples for

    Returns:
        Number of examples deleted
    """
    try:
        result = await db.code_examples.delete_many({"source_id": source_id})
        deleted_count = result.deleted_count

        search_logger.info(f"Deleted {deleted_count} code examples for source {source_id}")
        return deleted_count

    except Exception as e:
        search_logger.error(f"Error deleting code examples for source {source_id}: {e}")
        return 0


async def generate_code_summaries_batch(
    code_blocks: list[dict[str, Any]], max_workers: int = None, progress_callback=None
) -> list[dict[str, str]]:
    """
    Generate summaries for multiple code blocks with rate limiting and proper worker management.

    Args:
        code_blocks: List of code block dictionaries
        max_workers: Maximum number of concurrent API requests
        progress_callback: Optional callback for progress updates (async function)

    Returns:
        List of summary dictionaries
    """
    if not code_blocks:
        return []

    # Get max_workers from settings if not provided
    if max_workers is None:
        try:
            from ...services.credential_service import credential_service
            max_workers = int(await credential_service.get_credential("CODE_SUMMARIES_MAX_WORKERS", "3"))
        except Exception:
            max_workers = 3

    # For now, return default summaries since this requires LLM integration
    # TODO: Implement LLM-based code summarization
    results = []
    total_blocks = len(code_blocks)

    for i, code_block in enumerate(code_blocks):
        try:
            # Generate a simple summary based on code analysis
            code = code_block.get("code", "")
            language = code_block.get("language", "unknown")

            # Simple heuristic-based summary
            lines = len(code.split('\n'))
            chars = len(code)

            summary = f"Code block in {language} ({lines} lines, {chars} characters)"

            # Look for function/class definitions
            if 'function' in code.lower():
                summary += " - Contains function definitions"
            if 'class' in code.lower():
                summary += " - Contains class definitions"
            if 'import' in code.lower() or 'from' in code.lower():
                summary += " - Contains imports"

            results.append({
                "code": code,
                "summary": summary,
                "language": language,
            })

            # Report progress
            if progress_callback:
                progress = int(((i + 1) / total_blocks) * 100)
                await progress_callback({"percentage": progress})

        except Exception as e:
            search_logger.error(f"Error generating summary for code block: {e}")
            results.append({
                "code": code_block.get("code", ""),
                "summary": "Error generating summary",
                "language": code_block.get("language", "unknown"),
            })

    return results
