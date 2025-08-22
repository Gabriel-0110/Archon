"""
MongoDB Search Strategies

Implements vector similarity search and hybrid search using MongoDB Atlas Vector Search.
This replaces the Supabase-based search functionality.
"""

import math
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ...config.logfire_config import get_logger, safe_span

logger = get_logger(__name__)

# Fixed similarity threshold for vector results
SIMILARITY_THRESHOLD = 0.15


class MongoDBBaseSearchStrategy:
    """Base strategy implementing fundamental vector similarity search using MongoDB."""

    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize with MongoDB database."""
        self.db = db

    async def ensure_vector_search_index(self, collection_name: str = "documents"):
        """
        Ensure vector search index exists for MongoDB Atlas Vector Search.
        
        Note: This requires MongoDB Atlas (not self-hosted MongoDB).
        For self-hosted MongoDB, you'll need to use a different approach like
        calculating cosine similarity in aggregation pipelines.
        """
        try:
            collection = self.db[collection_name]

            # Check if vector search index exists
            indexes = await collection.list_indexes().to_list(length=None)
            vector_index_exists = any(
                idx.get("name") == "vector_index" for idx in indexes
            )

            if not vector_index_exists:
                # Create vector search index (Atlas Vector Search)
                # Note: This is specific to MongoDB Atlas
                index_spec = {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": 1536,  # OpenAI embedding dimensions
                                "similarity": "cosine"
                            }
                        }
                    }
                }

                logger.info(f"Vector search index would be created for {collection_name}")
                logger.warning(
                    "MongoDB Atlas Vector Search index creation requires Atlas cluster. "
                    "For self-hosted MongoDB, implement cosine similarity in aggregation pipeline."
                )

        except Exception as e:
            logger.warning(f"Could not ensure vector search index: {e}")

    def _calculate_cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        This is a fallback for non-Atlas MongoDB installations.
        """
        try:
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))

            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))

            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = dot_product / (magnitude1 * magnitude2)
            return similarity

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    async def vector_search(
        self,
        query_embedding: list[float],
        match_count: int,
        filter_metadata: dict | None = None,
        collection_name: str = "documents",
    ) -> list[dict[str, Any]]:
        """
        Perform vector similarity search using MongoDB.

        Args:
            query_embedding: The embedding vector for the query
            match_count: Number of results to return
            filter_metadata: Optional metadata filters
            collection_name: MongoDB collection to search

        Returns:
            List of matching documents with similarity scores
        """
        with safe_span("mongodb_vector_search", collection=collection_name, match_count=match_count) as span:
            try:
                collection = self.db[collection_name]

                # Try Atlas Vector Search first
                try:
                    # MongoDB Atlas Vector Search
                    pipeline = [
                        {
                            "$vectorSearch": {
                                "index": "vector_index",
                                "path": "embedding",
                                "queryVector": query_embedding,
                                "numCandidates": match_count * 10,
                                "limit": match_count
                            }
                        },
                        {
                            "$addFields": {
                                "similarity": {"$meta": "vectorSearchScore"}
                            }
                        }
                    ]

                    # Add metadata filters if provided
                    if filter_metadata:
                        match_stage = {"$match": {}}

                        if "source" in filter_metadata:
                            match_stage["$match"]["source_id"] = filter_metadata["source"]

                        # Add other metadata filters
                        for key, value in filter_metadata.items():
                            if key != "source":
                                match_stage["$match"][f"metadata.{key}"] = value

                        if match_stage["$match"]:
                            pipeline.append(match_stage)

                    # Execute Atlas Vector Search
                    cursor = collection.aggregate(pipeline)
                    results = await cursor.to_list(length=None)

                    # Filter by similarity threshold
                    filtered_results = []
                    for result in results:
                        similarity = float(result.get("similarity", 0.0))
                        if similarity >= SIMILARITY_THRESHOLD:
                            # Convert ObjectId to string
                            result["_id"] = str(result["_id"])
                            filtered_results.append(result)

                    span.set_attribute("search_method", "atlas_vector_search")
                    span.set_attribute("results_found", len(filtered_results))

                    return filtered_results

                except Exception as atlas_error:
                    logger.warning(f"Atlas Vector Search failed, falling back to cosine similarity: {atlas_error}")

                    # Fallback: Manual cosine similarity calculation
                    # This works with self-hosted MongoDB but is less efficient

                    # Build match criteria
                    match_criteria = {}
                    if filter_metadata:
                        if "source" in filter_metadata:
                            match_criteria["source_id"] = filter_metadata["source"]

                        for key, value in filter_metadata.items():
                            if key != "source":
                                match_criteria[f"metadata.{key}"] = value

                    # Get all documents (with limit for performance)
                    cursor = collection.find(match_criteria).limit(match_count * 50)
                    documents = await cursor.to_list(length=None)

                    # Calculate similarities manually
                    results_with_similarity = []
                    for doc in documents:
                        doc_embedding = doc.get("embedding")
                        if doc_embedding and len(doc_embedding) == len(query_embedding):
                            similarity = self._calculate_cosine_similarity(query_embedding, doc_embedding)
                            if similarity >= SIMILARITY_THRESHOLD:
                                doc["similarity"] = similarity
                                doc["_id"] = str(doc["_id"])
                                results_with_similarity.append(doc)

                    # Sort by similarity and limit results
                    results_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)
                    filtered_results = results_with_similarity[:match_count]

                    span.set_attribute("search_method", "manual_cosine_similarity")
                    span.set_attribute("results_found", len(filtered_results))

                    return filtered_results

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                span.set_attribute("error", str(e))
                return []

    async def search_code_examples(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search code examples using vector similarity.

        Args:
            query_embedding: The embedding vector for the query
            match_count: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of matching code examples with similarity scores
        """
        return await self.vector_search(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata,
            collection_name="code_examples"
        )


class MongoDBHybridSearchStrategy(MongoDBBaseSearchStrategy):
    """Hybrid search strategy combining vector search with text search using MongoDB."""

    async def search_documents_hybrid(
        self,
        query: str,
        query_embedding: list[float],
        match_count: int,
        filter_metadata: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and text search.

        Args:
            query: Original text query
            query_embedding: The embedding vector for the query
            match_count: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of matching documents with combined scores
        """
        with safe_span("mongodb_hybrid_search", match_count=match_count) as span:
            try:
                # Get vector search results
                vector_results = await self.vector_search(
                    query_embedding=query_embedding,
                    match_count=match_count * 2,  # Get more for combining
                    filter_metadata=filter_metadata,
                )

                # Get text search results
                text_results = await self._text_search(
                    query=query,
                    match_count=match_count * 2,
                    filter_metadata=filter_metadata,
                )

                # Combine and rerank results
                combined_results = self._combine_search_results(
                    vector_results, text_results, match_count
                )

                span.set_attribute("vector_results", len(vector_results))
                span.set_attribute("text_results", len(text_results))
                span.set_attribute("combined_results", len(combined_results))

                return combined_results

            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
                span.set_attribute("error", str(e))
                return []

    async def _text_search(
        self,
        query: str,
        match_count: int,
        filter_metadata: dict | None = None,
        collection_name: str = "documents",
    ) -> list[dict[str, Any]]:
        """
        Perform text search using MongoDB text indexes.

        Args:
            query: Text query
            match_count: Number of results to return
            filter_metadata: Optional metadata filters
            collection_name: MongoDB collection to search

        Returns:
            List of matching documents with text scores
        """
        try:
            collection = self.db[collection_name]

            # Ensure text index exists
            try:
                await collection.create_index([("content", "text"), ("metadata", "text")])
            except Exception:
                pass  # Index might already exist

            # Build search criteria
            search_criteria = {"$text": {"$search": query}}

            # Add metadata filters
            if filter_metadata:
                if "source" in filter_metadata:
                    search_criteria["source_id"] = filter_metadata["source"]

                for key, value in filter_metadata.items():
                    if key != "source":
                        search_criteria[f"metadata.{key}"] = value

            # Execute text search
            cursor = collection.find(
                search_criteria,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(match_count)

            results = await cursor.to_list(length=None)

            # Convert ObjectId to string and add text similarity
            for result in results:
                result["_id"] = str(result["_id"])
                result["text_similarity"] = result.get("score", 0.0)

            return results

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    def _combine_search_results(
        self,
        vector_results: list[dict[str, Any]],
        text_results: list[dict[str, Any]],
        match_count: int,
    ) -> list[dict[str, Any]]:
        """
        Combine vector and text search results with weighted scoring.

        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            match_count: Final number of results to return

        Returns:
            Combined and reranked results
        """
        # Create a map to combine results by document ID
        result_map = {}

        # Add vector results
        for result in vector_results:
            doc_id = result["_id"]
            result_map[doc_id] = result.copy()
            result_map[doc_id]["vector_similarity"] = result.get("similarity", 0.0)
            result_map[doc_id]["text_similarity"] = 0.0

        # Add text results
        for result in text_results:
            doc_id = result["_id"]
            if doc_id in result_map:
                # Update existing entry
                result_map[doc_id]["text_similarity"] = result.get("text_similarity", 0.0)
            else:
                # Add new entry
                result_map[doc_id] = result.copy()
                result_map[doc_id]["vector_similarity"] = 0.0
                result_map[doc_id]["text_similarity"] = result.get("text_similarity", 0.0)

        # Calculate combined scores
        for doc_id, result in result_map.items():
            vector_score = result.get("vector_similarity", 0.0)
            text_score = result.get("text_similarity", 0.0)

            # Weighted combination (70% vector, 30% text)
            combined_score = (0.7 * vector_score) + (0.3 * min(text_score / 10.0, 1.0))
            result["similarity"] = combined_score

        # Sort by combined score and return top results
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        return combined_results[:match_count]
