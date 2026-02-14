"""
Embedding Service using Sentence-Transformers (SBERT).

Provides text embedding functionality using the all-MiniLM-L6-v2 model.
This model produces 384-dimensional embeddings optimized for semantic similarity.
"""

from typing import List, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """
    Service for generating text embeddings using SBERT.
    
    Uses sentence-transformers/all-MiniLM-L6-v2 model which:
    - Produces 384-dimensional embeddings
    - Is optimized for semantic similarity tasks
    - Has fast inference speed
    - Works well for English text
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding model."""
        if EmbeddingService._model is None:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the SBERT model."""
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            EmbeddingService._model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the loaded model."""
        if EmbeddingService._model is None:
            self._load_model()
        return EmbeddingService._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return settings.embedding_dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed.
            
        Returns:
            Numpy array of shape (384,) containing the embedding.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.dimension, dtype=np.float32)
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            Numpy array of shape (n_texts, 384) containing embeddings.
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        # Filter out empty texts but keep track of indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        logger.debug(f"Generating embeddings for {len(valid_texts)} texts")
        
        # Generate embeddings for valid texts
        embeddings = self.model.encode(
            valid_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(valid_texts) > 100,
            batch_size=32,
        ).astype(np.float32)
        
        # Create result array with zeros for empty texts
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            result[idx] = embeddings[i]
        
        logger.debug(f"Generated embeddings with shape: {result.shape}")
        return result
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Cosine similarity score between -1 and 1.
        """
        # Since embeddings are already normalized, dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))
    
    def compute_similarities(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarities between a query and corpus embeddings.
        
        Args:
            query_embedding: Query embedding of shape (384,).
            corpus_embeddings: Corpus embeddings of shape (n, 384).
            
        Returns:
            Array of similarity scores of shape (n,).
        """
        # Dot product with normalized vectors = cosine similarity
        return np.dot(corpus_embeddings, query_embedding)


# Global instance for easy import
embedding_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    """Get the embedding service instance."""
    return embedding_service
