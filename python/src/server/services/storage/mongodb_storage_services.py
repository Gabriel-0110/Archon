"""
MongoDB Storage Services

This module contains all storage service classes that handle document and data storage operations
using MongoDB as the backend database.
"""

from datetime import datetime
from typing import Any

from bson import ObjectId
from fastapi import WebSocket

from ...config.logfire_config import get_logger, safe_span
from .base_storage_service import BaseStorageService
from .mongodb_document_storage import add_documents_to_mongodb

logger = get_logger(__name__)


class MongoDBDocumentStorageService(BaseStorageService):
    """Service for handling document uploads with progress reporting using MongoDB."""

    async def upload_document(
        self,
        file_content: str,
        filename: str,
        source_id: str,
        knowledge_type: str = "documentation",
        tags: list[str] | None = None,
        websocket: WebSocket | None = None,
        progress_callback: Any | None = None,
        cancellation_check: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Upload and process a document file with progress reporting.

        Args:
            file_content: Document content as text
            filename: Name of the file
            source_id: Source identifier
            knowledge_type: Type of knowledge
            tags: Optional list of tags
            websocket: Optional WebSocket for progress
            progress_callback: Optional callback for progress

        Returns:
            Tuple of (success, result_dict)
        """
        logger.info(f"Document upload starting: {filename} as {knowledge_type} knowledge")
        
        with safe_span(
            "upload_document",
            filename=filename,
            source_id=source_id,
            content_length=len(file_content),
        ) as span:
            try:
                # Progress reporting helper
                async def report_progress(message: str, percentage: int, batch_info: dict = None):
                    if websocket:
                        data = {
                            "type": "upload_progress",
                            "filename": filename,
                            "progress": percentage,
                            "message": message,
                        }
                        if batch_info:
                            data.update(batch_info)
                        await websocket.send_json(data)
                    if progress_callback:
                        await progress_callback(message, percentage, batch_info)

                await report_progress("Starting document processing...", 10)

                # Use base class chunking
                chunks = await self.smart_chunk_text_async(
                    file_content,
                    chunk_size=5000,
                    progress_callback=lambda msg, pct: report_progress(
                        f"Chunking: {msg}", 10 + float(pct) * 0.2
                    ),
                )

                if not chunks:
                    raise ValueError("No content could be extracted from the document")

                await report_progress("Preparing document chunks...", 30)

                # Prepare data for storage
                doc_url = f"file://{filename}"
                urls = []
                chunk_numbers = []
                contents = []
                metadatas = []
                total_word_count = 0

                # Process chunks with metadata
                for i, chunk in enumerate(chunks):
                    # Use base class metadata extraction
                    meta = self.extract_metadata(
                        chunk,
                        {
                            "chunk_index": i,
                            "url": doc_url,
                            "source": source_id,
                            "source_id": source_id,
                            "knowledge_type": knowledge_type,
                            "source_type": "file",
                            "filename": filename,
                        },
                    )

                    if tags:
                        meta["tags"] = tags

                    urls.append(doc_url)
                    chunk_numbers.append(i)
                    contents.append(chunk)
                    metadatas.append(meta)
                    total_word_count += meta.get("word_count", 0)

                await report_progress("Updating source information...", 50)

                # Create URL to full document mapping
                url_to_full_document = {doc_url: file_content}

                # Update source information in MongoDB
                await self._update_source_info_mongodb(
                    source_id, file_content, total_word_count, knowledge_type, tags
                )

                await report_progress("Storing document chunks...", 70)

                # Store documents in MongoDB
                await add_documents_to_mongodb(
                    db=self.db,
                    urls=urls,
                    chunk_numbers=chunk_numbers,
                    contents=contents,
                    metadatas=metadatas,
                    url_to_full_document=url_to_full_document,
                    batch_size=15,
                    progress_callback=progress_callback,
                    enable_parallel_batches=True,
                    provider=None,
                    cancellation_check=cancellation_check,
                )

                await report_progress("Document upload completed!", 100)

                result = {
                    "chunks_stored": len(chunks),
                    "total_word_count": total_word_count,
                    "source_id": source_id,
                    "filename": filename,
                }

                span.set_attribute("success", True)
                span.set_attribute("chunks_stored", len(chunks))
                span.set_attribute("total_word_count", total_word_count)

                logger.info(
                    f"Document upload completed successfully: filename={filename}, chunks_stored={len(chunks)}, total_word_count={total_word_count}"
                )

                return True, result

            except Exception as e:
                span.set_attribute("success", False)
                span.set_attribute("error", str(e))
                logger.error(f"Error uploading document: {e}")

                if websocket:
                    await websocket.send_json({
                        "type": "upload_error",
                        "error": str(e),
                        "filename": filename,
                    })

                return False, {"error": f"Error uploading document: {str(e)}"}

    async def _update_source_info_mongodb(
        self,
        source_id: str,
        file_content: str,
        total_word_count: int,
        knowledge_type: str,
        tags: list[str] | None = None,
    ):
        """Update source information in MongoDB."""
        try:
            # Extract source summary (first 5000 chars)
            from ...utils import extract_source_summary

            source_summary = await self.threading_service.run_cpu_intensive(
                extract_source_summary, source_id, file_content[:5000]
            )

            # Generate title from content (first 1000 chars)
            title = file_content[:100].strip().split('\n')[0] if file_content else source_id

            current_time = datetime.utcnow()

            # Create or update source document
            source_doc = {
                "source_id": source_id,
                "title": title,
                "summary": source_summary,
                "word_count": total_word_count,
                "knowledge_type": knowledge_type,
                "source_type": "file",
                "tags": tags or [],
                "updated_at": current_time,
            }

            # Upsert source information
            await self.db.sources.update_one(
                {"source_id": source_id},
                {
                    "$set": source_doc,
                    "$setOnInsert": {"created_at": current_time}
                },
                upsert=True
            )

            logger.info(f"Updated source info for {source_id} with knowledge_type={knowledge_type}")

        except Exception as e:
            logger.error(f"Error updating source info for {source_id}: {e}")
            # Don't fail the entire upload for source info errors
            pass

    async def store_documents(self, documents: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Store multiple documents. Implementation of abstract method.

        Args:
            documents: List of documents to store
            **kwargs: Additional options (websocket, progress_callback, etc.)

        Returns:
            Storage result
        """
        results = []
        for doc in documents:
            success, result = await self.upload_document(
                file_content=doc["content"],
                filename=doc["filename"],
                source_id=doc.get("source_id", "upload"),
                knowledge_type=doc.get("knowledge_type", "documentation"),
                tags=doc.get("tags"),
                websocket=kwargs.get("websocket"),
                progress_callback=kwargs.get("progress_callback"),
                cancellation_check=kwargs.get("cancellation_check"),
            )
            results.append(result)

        return {
            "success": all(r.get("chunks_stored", 0) > 0 for r in results),
            "documents_processed": len(documents),
            "results": results,
        }

    async def process_document(self, document: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Process a single document. Implementation of abstract method.

        Args:
            document: Document to process
            **kwargs: Additional processing options

        Returns:
            Processed document with metadata
        """
        # Extract text content
        content = document.get("content", "")

        # Chunk the content
        chunks = await self.smart_chunk_text_async(content)

        # Extract metadata for each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            meta = self.extract_metadata(
                chunk, {"chunk_index": i, "source": document.get("source", "unknown")}
            )
            processed_chunks.append({"content": chunk, "metadata": meta})

        return {
            "chunks": processed_chunks,
            "total_chunks": len(chunks),
            "source": document.get("source"),
        }

    async def store_code_examples(
        self, code_examples: list[dict[str, Any]]
    ) -> tuple[bool, dict[str, Any]]:
        """
        Store code examples in MongoDB.

        Args:
            code_examples: List of code examples

        Returns:
            Tuple of (success, result)
        """
        try:
            if not code_examples:
                return True, {"code_examples_stored": 0}

            current_time = datetime.utcnow()
            
            # Prepare code examples for MongoDB
            mongo_examples = []
            for example in code_examples:
                mongo_example = {
                    "_id": ObjectId(),
                    "source_id": example.get("source_id"),
                    "url": example.get("url"),
                    "language": example.get("language"),
                    "code": example.get("code"),
                    "context": example.get("context", ""),
                    "line_start": example.get("line_start"),
                    "line_end": example.get("line_end"),
                    "metadata": example.get("metadata", {}),
                    "created_at": current_time,
                    "updated_at": current_time,
                }
                mongo_examples.append(mongo_example)

            # Insert code examples
            result = await self.db.code_examples.insert_many(mongo_examples)
            
            logger.info(f"Stored {len(result.inserted_ids)} code examples")
            
            return True, {"code_examples_stored": len(result.inserted_ids)}

        except Exception as e:
            logger.error(f"Error storing code examples: {e}")
            return False, {"error": str(e)}