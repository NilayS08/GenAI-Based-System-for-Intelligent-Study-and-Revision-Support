from fastapi import APIRouter, UploadFile, File, HTTPException, status
from uuid import uuid4

from app.services.file_handler import extract_text_from_file
from app.services.preprocessing import clean_text
from app.database import supabase

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT),
    extract text, clean it, and store in Supabase.
    """

    try:
        # Extract raw text
        raw_text = await extract_text_from_file(file)

        # Clean text
        cleaned_text = clean_text(raw_text)

        if not cleaned_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file contains no readable text."
            )

        document_id = str(uuid4())

        # Store in Supabase
        result = supabase.table("documents").insert({
            "id": document_id,
            "filename": file.filename,
            "content": cleaned_text
        }).execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store document in database."
            )

        return {
            "message": "Document uploaded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "content_length": len(cleaned_text)
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
