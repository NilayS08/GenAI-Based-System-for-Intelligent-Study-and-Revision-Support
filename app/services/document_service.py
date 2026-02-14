"""
Document Service for Supabase Operations.

Handles all database operations for the documents table including:
- Insert new documents
- Retrieve documents
- Update documents
- Delete documents

Uses Supabase PostgreSQL as the backend.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from loguru import logger
from postgrest.exceptions import APIError

from app.database import supabase
from app.models.schemas import DocumentCreate, DocumentResponse


class DocumentService:
    """
    Service class for document-related Supabase operations.
    
    Handles CRUD operations for the documents table.
    All embeddings are stored in FAISS, not Supabase.
    """
    
    TABLE_NAME = "documents"
    
    def __init__(self):
        """Initialize the document service."""
        self.client = supabase
        logger.info("DocumentService initialized")
    
    async def insert_document(
        self,
        filename: str,
        file_size: Optional[int] = None,
        file_type: Optional[str] = None,
    ) -> Optional[DocumentResponse]:
        """
        Insert a new document record into Supabase.
        
        Args:
            filename: Name of the uploaded file
            file_size: Size of the file in bytes (optional)
            file_type: MIME type or extension of the file (optional)
        
        Returns:
            DocumentResponse with the created document data, or None on failure
        
        Raises:
            APIError: If Supabase insert fails
        """
        try:
            document_data = {
                "filename": filename,
                "upload_timestamp": datetime.utcnow().isoformat(),
            }
            
            # Add optional fields if provided
            if file_size is not None:
                document_data["file_size"] = file_size
            if file_type is not None:
                document_data["file_type"] = file_type
            
            logger.info(f"Inserting document: {filename}")
            
            response = self.client.table(self.TABLE_NAME).insert(document_data).execute()
            
            if response.data:
                document = response.data[0]
                logger.info(f"Document inserted successfully with ID: {document['id']}")
                
                return DocumentResponse(
                    id=UUID(document["id"]),
                    filename=document["filename"],
                    upload_timestamp=datetime.fromisoformat(
                        document["upload_timestamp"].replace("Z", "+00:00")
                    ),
                    file_size=document.get("file_size"),
                    file_type=document.get("file_type"),
                )
            
            logger.warning("Insert returned no data")
            return None
            
        except APIError as e:
            logger.error(f"Supabase API error inserting document: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error inserting document: {e}")
            raise
    
    async def insert_document_from_schema(
        self,
        document: DocumentCreate,
        file_size: Optional[int] = None,
        file_type: Optional[str] = None,
    ) -> Optional[DocumentResponse]:
        """
        Insert a document using Pydantic schema.
        
        Args:
            document: DocumentCreate schema with document data
            file_size: Size of the file in bytes (optional)
            file_type: MIME type or extension of the file (optional)
        
        Returns:
            DocumentResponse with the created document data
        """
        return await self.insert_document(
            filename=document.filename,
            file_size=file_size,
            file_type=file_type,
        )
    
    async def bulk_insert_documents(
        self,
        documents: List[DocumentCreate],
    ) -> List[DocumentResponse]:
        """
        Insert multiple documents in a single operation.
        
        Args:
            documents: List of DocumentCreate schemas
        
        Returns:
            List of DocumentResponse with created documents
        
        Note:
            This is more efficient than inserting one by one.
        """
        try:
            documents_data = [
                {
                    "filename": doc.filename,
                    "upload_timestamp": datetime.utcnow().isoformat(),
                }
                for doc in documents
            ]
            
            logger.info(f"Bulk inserting {len(documents)} documents")
            
            response = self.client.table(self.TABLE_NAME).insert(documents_data).execute()
            
            if response.data:
                logger.info(f"Successfully inserted {len(response.data)} documents")
                
                return [
                    DocumentResponse(
                        id=UUID(doc["id"]),
                        filename=doc["filename"],
                        upload_timestamp=datetime.fromisoformat(
                            doc["upload_timestamp"].replace("Z", "+00:00")
                        ),
                        file_size=doc.get("file_size"),
                        file_type=doc.get("file_type"),
                    )
                    for doc in response.data
                ]
            
            return []
            
        except APIError as e:
            logger.error(f"Supabase API error in bulk insert: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in bulk insert: {e}")
            raise
    
    async def get_document_by_id(self, document_id: UUID) -> Optional[DocumentResponse]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: UUID of the document
        
        Returns:
            DocumentResponse if found, None otherwise
        """
        try:
            logger.debug(f"Fetching document with ID: {document_id}")
            
            response = (
                self.client.table(self.TABLE_NAME)
                .select("*")
                .eq("id", str(document_id))
                .execute()
            )
            
            if response.data:
                doc = response.data[0]
                return DocumentResponse(
                    id=UUID(doc["id"]),
                    filename=doc["filename"],
                    upload_timestamp=datetime.fromisoformat(
                        doc["upload_timestamp"].replace("Z", "+00:00")
                    ),
                    file_size=doc.get("file_size"),
                    file_type=doc.get("file_type"),
                )
            
            logger.warning(f"Document not found with ID: {document_id}")
            return None
            
        except APIError as e:
            logger.error(f"Supabase API error fetching document: {e}")
            raise
    
    async def get_document_by_filename(self, filename: str) -> Optional[DocumentResponse]:
        """
        Retrieve a document by its filename.
        
        Args:
            filename: Name of the file
        
        Returns:
            DocumentResponse if found, None otherwise
        """
        try:
            logger.debug(f"Fetching document with filename: {filename}")
            
            response = (
                self.client.table(self.TABLE_NAME)
                .select("*")
                .eq("filename", filename)
                .order("upload_timestamp", desc=True)
                .limit(1)
                .execute()
            )
            
            if response.data:
                doc = response.data[0]
                return DocumentResponse(
                    id=UUID(doc["id"]),
                    filename=doc["filename"],
                    upload_timestamp=datetime.fromisoformat(
                        doc["upload_timestamp"].replace("Z", "+00:00")
                    ),
                    file_size=doc.get("file_size"),
                    file_type=doc.get("file_type"),
                )
            
            return None
            
        except APIError as e:
            logger.error(f"Supabase API error fetching document by filename: {e}")
            raise
    
    async def get_all_documents(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentResponse]:
        """
        Retrieve all documents with pagination.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
        
        Returns:
            List of DocumentResponse
        """
        try:
            logger.debug(f"Fetching documents with limit={limit}, offset={offset}")
            
            response = (
                self.client.table(self.TABLE_NAME)
                .select("*")
                .order("upload_timestamp", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )
            
            return [
                DocumentResponse(
                    id=UUID(doc["id"]),
                    filename=doc["filename"],
                    upload_timestamp=datetime.fromisoformat(
                        doc["upload_timestamp"].replace("Z", "+00:00")
                    ),
                    file_size=doc.get("file_size"),
                    file_type=doc.get("file_type"),
                )
                for doc in response.data
            ]
            
        except APIError as e:
            logger.error(f"Supabase API error fetching all documents: {e}")
            raise
    
    async def delete_document(self, document_id: UUID) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            document_id: UUID of the document to delete
        
        Returns:
            True if deletion was successful, False otherwise
        
        Note:
            This only deletes the metadata from Supabase.
            FAISS embeddings must be handled separately.
        """
        try:
            logger.info(f"Deleting document with ID: {document_id}")
            
            response = (
                self.client.table(self.TABLE_NAME)
                .delete()
                .eq("id", str(document_id))
                .execute()
            )
            
            if response.data:
                logger.info(f"Document deleted successfully: {document_id}")
                return True
            
            logger.warning(f"No document found to delete with ID: {document_id}")
            return False
            
        except APIError as e:
            logger.error(f"Supabase API error deleting document: {e}")
            raise
    
    async def check_document_exists(self, filename: str) -> bool:
        """
        Check if a document with the given filename already exists.
        
        Args:
            filename: Name of the file to check
        
        Returns:
            True if document exists, False otherwise
        """
        try:
            response = (
                self.client.table(self.TABLE_NAME)
                .select("id")
                .eq("filename", filename)
                .limit(1)
                .execute()
            )
            
            return len(response.data) > 0
            
        except APIError as e:
            logger.error(f"Supabase API error checking document existence: {e}")
            raise


# Singleton instance for dependency injection
document_service = DocumentService()


def get_document_service() -> DocumentService:
    """
    Get the document service instance.
    
    Used for FastAPI dependency injection.
    
    Returns:
        DocumentService singleton instance
    """
    return document_service
