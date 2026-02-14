"""
Smart Revision Generator - FastAPI Application Entry Point

GenAI-Based Intelligent Study and Revision Support System
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings


# Global references for initialized services
_embedding_service = None
_vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    global _embedding_service, _vector_store
    
    # Startup
    logger.info(f"Starting {settings.app_name}...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize SBERT Embedding Service
    logger.info("Initializing SBERT embedding model...")
    from app.services.embedding import get_embedding_service
    _embedding_service = get_embedding_service()
    logger.info(f"SBERT model loaded: {settings.embedding_model}")
    logger.info(f"Embedding dimension: {_embedding_service.dimension}")
    
    # Initialize FAISS Vector Store
    logger.info("Initializing FAISS vector store...")
    from app.vector_store import initialize_vector_store
    _vector_store = initialize_vector_store()
    logger.info(f"FAISS index initialized with {_vector_store.size} vectors")
    
    # TODO: Initialize Supabase client
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Save FAISS index on shutdown
    if _vector_store and _vector_store.size > 0:
        logger.info("Saving FAISS index...")
        _vector_store.save()
        logger.info("FAISS index saved successfully")


# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="GenAI-Based Intelligent Study and Revision Support System using RAG and Hybrid Retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "message": "Smart Revision Generator API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "services": {
            "api": "running",
            "embedding_model": settings.embedding_model if _embedding_service else "not initialized",
            "embedding_dimension": _embedding_service.dimension if _embedding_service else 0,
            "vector_store": f"active ({_vector_store.size} vectors)" if _vector_store else "not initialized",
            "database": "pending",  # TODO: Check Supabase connection
        },
    }


# TODO: Include routers
# from app.routers import documents, generations, feedback
# app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
# app.include_router(generations.router, prefix="/api/v1/generations", tags=["Generations"])
# app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
