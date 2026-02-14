"""
Chunking Service for Text Segmentation.

Implements token-based chunking with overlap for RAG pipeline.
Uses tiktoken for accurate token counting compatible with LLMs.

Chunking Strategy:
- Target chunk size: 300-500 tokens (configurable via settings)
- Overlap: 50 tokens (configurable) to maintain context continuity
- Sentence-aware splitting to avoid cutting mid-sentence
- Preserves semantic coherence within chunks
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from uuid import UUID, uuid4

import tiktoken
from loguru import logger

from app.config import settings


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: UUID
    text: str
    index: int
    token_count: int
    start_char: int  # Starting character position in original text
    end_char: int    # Ending character position in original text
    document_id: Optional[UUID] = None


class ChunkingService:
    """
    Service for chunking text into overlapping segments.
    
    Uses tiktoken for accurate token counting and implements
    sentence-aware splitting to maintain context quality.
    
    Configuration (from settings):
        - chunk_size: Target tokens per chunk (default: 400)
        - chunk_overlap: Overlap tokens between chunks (default: 50)
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        encoding_name: str = "cl100k_base",  # GPT-4, GPT-3.5-turbo encoding
    ):
        """
        Initialize the chunking service.
        
        Args:
            chunk_size: Target tokens per chunk (uses settings if None)
            chunk_overlap: Overlap tokens between chunks (uses settings if None)
            encoding_name: tiktoken encoding name for token counting
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Validate parameters
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
            logger.info(
                f"ChunkingService initialized: size={self.chunk_size}, "
                f"overlap={self.chunk_overlap}, encoding={encoding_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Handles common sentence endings while preserving:
        - Abbreviations (Dr., Mr., etc.)
        - Decimal numbers
        - URLs and emails
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Sentence splitting pattern
        # Splits on . ! ? followed by space and capital letter or end of string
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        
        # First, protect common abbreviations
        protected = text
        abbreviations = [
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.',
            'vs.', 'etc.', 'e.g.', 'i.e.', 'Fig.', 'fig.',
            'Vol.', 'vol.', 'No.', 'no.', 'pp.', 'p.'
        ]
        placeholder_map = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            placeholder_map[placeholder] = abbr
            protected = protected.replace(abbr, placeholder)
        
        # Split into sentences
        sentences = re.split(sentence_endings, protected)
        
        # Restore abbreviations and filter empty strings
        result = []
        for sent in sentences:
            for placeholder, abbr in placeholder_map.items():
                sent = sent.replace(placeholder, abbr)
            sent = sent.strip()
            if sent:
                result.append(sent)
        
        return result
    
    def _merge_sentences_to_chunks(
        self,
        sentences: List[str],
        document_id: Optional[UUID] = None,
    ) -> List[Chunk]:
        """
        Merge sentences into chunks respecting token limits and overlap.
        
        Args:
            sentences: List of sentences to merge
            document_id: Optional document ID to associate with chunks
            
        Returns:
            List of Chunk objects
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        chunk_index = 0
        char_position = 0
        chunk_start_char = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Handle sentences longer than chunk_size
            if sentence_tokens > self.chunk_size:
                # First, save current chunk if not empty
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(Chunk(
                        id=uuid4(),
                        text=chunk_text,
                        index=chunk_index,
                        token_count=current_token_count,
                        start_char=chunk_start_char,
                        end_char=char_position,
                        document_id=document_id,
                    ))
                    chunk_index += 1
                
                # Split long sentence into smaller parts
                long_chunks = self._split_long_text(
                    sentence, 
                    chunk_index, 
                    char_position,
                    document_id
                )
                chunks.extend(long_chunks)
                chunk_index += len(long_chunks)
                char_position += len(sentence) + 1
                
                # Reset for next chunk
                current_chunk_sentences = []
                current_token_count = 0
                chunk_start_char = char_position
                continue
            
            # Check if adding this sentence exceeds chunk_size
            potential_tokens = current_token_count + sentence_tokens
            if current_chunk_sentences:
                potential_tokens += 1  # Account for space between sentences
            
            if potential_tokens > self.chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(Chunk(
                    id=uuid4(),
                    text=chunk_text,
                    index=chunk_index,
                    token_count=current_token_count,
                    start_char=chunk_start_char,
                    end_char=char_position,
                    document_id=document_id,
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences, overlap_tokens = self._get_overlap_sentences(
                    current_chunk_sentences
                )
                current_chunk_sentences = overlap_sentences
                current_token_count = overlap_tokens
                chunk_start_char = char_position - sum(
                    len(s) + 1 for s in overlap_sentences
                )
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
            if len(current_chunk_sentences) > 1:
                current_token_count += 1  # Space token
            char_position += len(sentence) + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(Chunk(
                id=uuid4(),
                text=chunk_text,
                index=chunk_index,
                token_count=self.count_tokens(chunk_text),
                start_char=chunk_start_char,
                end_char=char_position,
                document_id=document_id,
            ))
        
        return chunks
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
    ) -> Tuple[List[str], int]:
        """
        Get sentences for overlap from the end of current chunk.
        
        Args:
            sentences: Current chunk's sentences
            
        Returns:
            Tuple of (overlap sentences, token count)
        """
        overlap_sentences = []
        overlap_tokens = 0
        
        # Work backwards from end of sentences
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences, overlap_tokens
    
    def _split_long_text(
        self,
        text: str,
        start_index: int,
        start_char: int,
        document_id: Optional[UUID] = None,
    ) -> List[Chunk]:
        """
        Split text longer than chunk_size into multiple chunks.
        
        Uses word boundaries to avoid splitting mid-word.
        
        Args:
            text: Long text to split
            start_index: Starting chunk index
            start_char: Starting character position
            document_id: Optional document ID
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        words = text.split()
        current_words = []
        current_tokens = 0
        chunk_index = start_index
        char_pos = start_char
        chunk_start = start_char
        
        for word in words:
            word_tokens = self.count_tokens(word)
            
            if current_tokens + word_tokens > self.chunk_size and current_words:
                # Save current chunk
                chunk_text = " ".join(current_words)
                chunks.append(Chunk(
                    id=uuid4(),
                    text=chunk_text,
                    index=chunk_index,
                    token_count=current_tokens,
                    start_char=chunk_start,
                    end_char=char_pos,
                    document_id=document_id,
                ))
                chunk_index += 1
                
                # Calculate overlap words
                overlap_words = []
                overlap_tokens = 0
                for w in reversed(current_words):
                    w_tokens = self.count_tokens(w)
                    if overlap_tokens + w_tokens <= self.chunk_overlap:
                        overlap_words.insert(0, w)
                        overlap_tokens += w_tokens
                    else:
                        break
                
                current_words = overlap_words
                current_tokens = overlap_tokens
                chunk_start = char_pos - sum(len(w) + 1 for w in overlap_words)
            
            current_words.append(word)
            current_tokens += word_tokens
            char_pos += len(word) + 1
        
        # Last chunk
        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append(Chunk(
                id=uuid4(),
                text=chunk_text,
                index=chunk_index,
                token_count=self.count_tokens(chunk_text),
                start_char=chunk_start,
                end_char=char_pos,
                document_id=document_id,
            ))
        
        return chunks
    
    def chunk_text(
        self,
        text: str,
        document_id: Optional[UUID] = None,
    ) -> List[Chunk]:
        """
        Chunk text into overlapping segments.
        
        Main entry point for chunking. Splits text into sentences,
        then merges sentences into chunks respecting token limits.
        
        Args:
            text: Input text to chunk
            document_id: Optional document ID to associate with chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Clean and normalize text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        logger.debug(f"Chunking text of {len(text)} characters, {self.count_tokens(text)} tokens")
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        logger.debug(f"Split into {len(sentences)} sentences")
        
        # Merge sentences into chunks
        chunks = self._merge_sentences_to_chunks(sentences, document_id)
        
        logger.info(
            f"Created {len(chunks)} chunks from text "
            f"(avg {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens/chunk)"
            if chunks else "Created 0 chunks"
        )
        
        return chunks
    
    def chunk_texts(
        self,
        texts: List[str],
        document_ids: Optional[List[UUID]] = None,
    ) -> List[List[Chunk]]:
        """
        Chunk multiple texts.
        
        Args:
            texts: List of texts to chunk
            document_ids: Optional list of document IDs (must match texts length)
            
        Returns:
            List of chunk lists, one per input text
        """
        if document_ids and len(document_ids) != len(texts):
            raise ValueError("document_ids length must match texts length")
        
        results = []
        for i, text in enumerate(texts):
            doc_id = document_ids[i] if document_ids else None
            chunks = self.chunk_text(text, doc_id)
            results.append(chunks)
        
        return results
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> dict:
        """
        Get statistics about a list of chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
            }
        
        token_counts = [c.token_count for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens": sum(token_counts) / len(chunks),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
        }


# Global instance for easy import
_chunking_service: Optional[ChunkingService] = None


def get_chunking_service() -> ChunkingService:
    """Get or create the global chunking service instance."""
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = ChunkingService()
    return _chunking_service


# Convenience function
def chunk_text(
    text: str,
    document_id: Optional[UUID] = None,
) -> List[Chunk]:
    """
    Convenience function to chunk text using the global service.
    
    Args:
        text: Input text to chunk
        document_id: Optional document ID
        
    Returns:
        List of Chunk objects
    """
    return get_chunking_service().chunk_text(text, document_id)
