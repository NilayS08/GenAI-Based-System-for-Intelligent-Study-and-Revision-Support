from fastapi import APIRouter, UploadFile, File
from app.services.preprocessing import clean_text
from app.database import supabase
from app.services.generation import generate_summary

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    cleaned = clean_text(text)

    supabase.table("documents").insert({
        "filename": file.filename,
        "content": cleaned
    }).execute()

    summary = generate_summary(cleaned[:4000])

    return {
        "message": "File uploaded successfully",
        "summary": summary
    }
