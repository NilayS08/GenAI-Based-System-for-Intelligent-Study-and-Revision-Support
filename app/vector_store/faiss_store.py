"""
FAISS Vector Store for Dense Retrieval.

Implements HNSW (Hierarchical Navigable Small World) index for efficient
approximate nearest neighbor search on embeddings.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import faiss
import numpy as np
from loguru import logger

from app.config import settings
from app.services.embedding import embedding_service


class FAISSVectorStore:
    """
    FAISS-based vector store for storing and searching embeddings.
    
    Uses HNSW index for efficient approximate nearest neighbor search.
    Supports:
    - Adding embeddings with metadata
    - Searching for similar vectors
    - Persistence (save/load from disk)
    """
    
    def __init__(
        self,
        dimension: int = None,
        index_path: str = None,
    ):
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: Embedding dimension (default: from settings).
            index_path: Path to save/load index (default: from settings).
        """
        self.dimension = dimension or settings.embedding_dimension
        self.index_path = index_path or settings.faiss_index_path
        
        # FAISS index
        self._index: Optional[faiss.Index] = None
        
        # Metadata storage: maps FAISS internal ID to chunk metadata
        # Structure: {faiss_id: {"chunk_id": UUID, "document_id": UUID, "chunk_text": str}}
        self._metadata: Dict[int, dict] = {}
        
        # Counter for internal IDs
        self._next_id: int = 0
        
        # Ensure directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def index(self) -> faiss.Index:
        """Get or create the FAISS index."""
        if self._index is None:
            self._create_index()
        return self._index
    
    def _create_index(self) -> None:
        """
        Create a new FAISS HNSW index.
        
        HNSW parameters:
        - M: Number of connections per layer (higher = better recall, more memory)
        - efConstruction: Search depth during construction (higher = better quality)
        - efSearch: Search depth during query (higher = better recall, slower)
        """
        logger.info(f"Creating new FAISS HNSW index with dimension {self.dimension}")
        
        # Create HNSW index with L2 distance
        # M=32 provides good balance between memory and recall
        M = 32
        self._index = faiss.IndexHNSWFlat(self.dimension, M)
        
        # Set construction-time search depth
        self._index.hnsw.efConstruction = 64
        
        # Set query-time search depth (can be changed later)
        self._index.hnsw.efSearch = 32
        
        logger.info("FAISS HNSW index created successfully")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[dict],
    ) -> List[int]:
        """
        Add embeddings with metadata to the index.
        
        Args:
            embeddings: Numpy array of shape (n, dimension).
            metadata_list: List of metadata dicts for each embedding.
                Each dict should have: chunk_id, document_id, chunk_text
                
        Returns:
            List of internal FAISS IDs assigned to the embeddings.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match metadata list length")
        
        if len(embeddings) == 0:
            return []
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Assign internal IDs
        ids = list(range(self._next_id, self._next_id + len(embeddings)))
        self._next_id += len(embeddings)
        
        # Store metadata
        for idx, meta in zip(ids, metadata_list):
            self._metadata[idx] = meta
        
        # Add to index
        self.index.add(embeddings)
        
        logger.info(f"Added {len(embeddings)} embeddings to FAISS index. Total: {self.index.ntotal}")
        
        return ids
    
    def add_texts(
        self,
        texts: List[str],
        metadata_list: List[dict],
    ) -> List[int]:
        """
        Add texts to the index (generates embeddings automatically).
        
        Args:
            texts: List of text strings to embed and add.
            metadata_list: List of metadata dicts for each text.
            
        Returns:
            List of internal FAISS IDs assigned.
        """
        # Generate embeddings
        embeddings = embedding_service.embed_texts(texts)
        
        # Add embeddings with metadata
        return self.add_embeddings(embeddings, metadata_list)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
    ) -> List[Tuple[int, float, dict]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding of shape (dimension,).
            top_k: Number of results to return (default: from settings).
            
        Returns:
            List of tuples: (faiss_id, distance, metadata)
            Sorted by distance (ascending for L2, so lower is better).
        """
        top_k = top_k or settings.top_k_results
        
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, returning no results")
            return []
        
        # Ensure query is 2D and float32
        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype(np.float32)
        )
        
        # Search
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))
        
        # Build results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            metadata = self._metadata.get(idx, {})
            results.append((idx, float(dist), metadata))
        
        return results
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = None,
    ) -> List[Tuple[int, float, dict]]:
        """
        Search by text query (generates embedding automatically).
        
        Args:
            query_text: Query string.
            top_k: Number of results to return.
            
        Returns:
            List of tuples: (faiss_id, distance, metadata).
        """
        query_embedding = embedding_service.embed_text(query_text)
        return self.search(query_embedding, top_k)
    
    def get_similarity_scores(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
    ) -> List[Tuple[int, float, dict]]:
        """
        Search and return cosine similarity scores instead of L2 distances.
        
        Since embeddings are L2-normalized, we convert L2 distance to cosine:
        cosine_sim = 1 - (L2_dist^2 / 2)
        
        Args:
            query_embedding: Query embedding.
            top_k: Number of results.
            
        Returns:
            List of tuples: (faiss_id, similarity_score, metadata).
            Sorted by similarity (descending, so higher is better).
        """
        results = self.search(query_embedding, top_k)
        
        # Convert L2 distance to cosine similarity
        # For normalized vectors: ||a - b||^2 = 2 - 2*cos(a,b)
        # Therefore: cos(a,b) = 1 - ||a - b||^2 / 2
        converted = []
        for faiss_id, l2_dist, metadata in results:
            cosine_sim = 1 - (l2_dist ** 2) / 2
            converted.append((faiss_id, cosine_sim, metadata))
        
        # Sort by similarity (descending)
        converted.sort(key=lambda x: x[1], reverse=True)
        
        return converted
    
    def save(self, path: str = None) -> None:
        """
        Save the index and metadata to disk.
        
        Args:
            path: Optional custom path (default: self.index_path).
        """
        path = path or self.index_path
        
        # Save FAISS index
        index_file = f"{path}.index"
        faiss.write_index(self.index, index_file)
        logger.info(f"Saved FAISS index to {index_file}")
        
        # Save metadata
        metadata_file = f"{path}.meta"
        with open(metadata_file, "wb") as f:
            pickle.dump({
                "metadata": self._metadata,
                "next_id": self._next_id,
            }, f)
        logger.info(f"Saved metadata to {metadata_file}")
    
    def load(self, path: str = None) -> bool:
        """
        Load the index and metadata from disk.
        
        Args:
            path: Optional custom path (default: self.index_path).
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        path = path or self.index_path
        index_file = f"{path}.index"
        metadata_file = f"{path}.meta"
        
        # Check if files exist
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            logger.warning(f"Index files not found at {path}")
            return False
        
        try:
            # Load FAISS index
            self._index = faiss.read_index(index_file)
            logger.info(f"Loaded FAISS index from {index_file} with {self._index.ntotal} vectors")
            
            # Load metadata
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)
                self._metadata = data["metadata"]
                self._next_id = data["next_id"]
            logger.info(f"Loaded metadata with {len(self._metadata)} entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def clear(self) -> None:
        """Clear the index and metadata."""
        self._index = None
        self._metadata = {}
        self._next_id = 0
        self._create_index()
        logger.info("Cleared FAISS index and metadata")
    
    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        return self.index.ntotal
    
    def get_metadata(self, faiss_id: int) -> Optional[dict]:
        """Get metadata for a specific FAISS ID."""
        return self._metadata.get(faiss_id)


# Global instance for easy import
_vector_store: Optional[FAISSVectorStore] = None


def get_vector_store() -> FAISSVectorStore:
    """
    Get or create the global vector store instance.
    Attempts to load from disk if exists.
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
        # Try to load existing index
        if not _vector_store.load():
            logger.info("No existing index found, using empty index")
    
    return _vector_store


def initialize_vector_store(force_new: bool = False) -> FAISSVectorStore:
    """
    Initialize the vector store.
    
    Args:
        force_new: If True, create a new index even if one exists.
        
    Returns:
        The initialized vector store.
    """
    global _vector_store
    
    if force_new:
        _vector_store = FAISSVectorStore()
        logger.info("Created new FAISS vector store")
    else:
        _vector_store = get_vector_store()
    
    return _vector_store
