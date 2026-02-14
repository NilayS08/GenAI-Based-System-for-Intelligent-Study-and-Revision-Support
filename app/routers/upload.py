from fastapi import APIRouter, UploadFile, File, HTTPException, status
from uuid import uuid4, UUID
from datetime import datetime

from app.services.file_handler import extract_text_from_file
from app.services.preprocessing import clean_text
from app.services.chunking import get_chunking_service
from app.services.embedding import get_embedding_service
from app.vector_store import get_vector_store
from app.database import supabase
from loguru import logger

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT),
    extract text, clean it, chunk it, generate embeddings,
    and store in Supabase + FAISS.
    
    Pipeline:
    1. Extract text from file
    2. Clean and preprocess text
    3. Store document metadata in Supabase (documents table)
    4. Chunk text into segments
    5. Store chunks in Supabase (chunks table)
    6. Generate embeddings and store in FAISS
    """

    try:
        # Step 1: Extract raw text
        logger.info(f"Processing upload: {file.filename}")
        raw_text = await extract_text_from_file(file)

        # Step 2: Clean text
        cleaned_text = clean_text(raw_text)

        if not cleaned_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file contains no readable text."
            )

        document_id = str(uuid4())
        
        # Step 3: Store document metadata in Supabase
        logger.info(f"Storing document metadata: {document_id}")
        doc_result = supabase.table("documents").insert({
            "id": document_id,
            "filename": file.filename,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "file_size": len(cleaned_text),
            "file_type": file.content_type or file.filename.split('.')[-1] if '.' in file.filename else "unknown"
        }).execute()

        if not doc_result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store document in database."
            )

        # Step 4: Chunk the text
        logger.info(f"Chunking document: {document_id}")
        chunking_service = get_chunking_service()
        chunks = chunking_service.chunk_text(cleaned_text, document_id=UUID(document_id))
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create chunks from document."
            )

        # Step 5: Store chunks in Supabase
        logger.info(f"Storing {len(chunks)} chunks in database")
        chunk_records = []
        for chunk in chunks:
            chunk_records.append({
                "id": str(chunk.id),
                "document_id": document_id,
                "chunk_text": chunk.text,
                "chunk_index": chunk.index,
                "token_count": chunk.token_count
            })
        
        chunk_result = supabase.table("chunks").insert(chunk_records).execute()
        
        if not chunk_result.data:
            # Rollback: delete document if chunks fail
            supabase.table("documents").delete().eq("id", document_id).execute()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store chunks in database."
            )

        # Step 6: Generate embeddings and store in FAISS
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()
        
        # Prepare texts and metadata for FAISS
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_metadata = [
            {
                "chunk_id": str(chunk.id),
                "document_id": document_id,
                "chunk_text": chunk.text,
                "chunk_index": chunk.index,
            }
            for chunk in chunks
        ]
        
        # Generate embeddings and add to FAISS
        embeddings = embedding_service.embed_texts(chunk_texts)
        vector_store.add_embeddings(embeddings, chunk_metadata)
        
        # Save FAISS index
        vector_store.save()
        logger.info(f"Document {document_id} processed successfully")

        return {
            "message": "Document uploaded and processed successfully",
            "document_id": document_id,
            "filename": file.filename,
            "content_length": len(cleaned_text),
            "chunks_created": len(chunks),
            "embeddings_stored": len(chunks)
        }

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Internal error processing upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
