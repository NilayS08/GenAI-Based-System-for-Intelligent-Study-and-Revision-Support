"""
Smart Revision Generator - FastAPI Application Entry Point

GenAI-Based Intelligent Study and Revision Support System
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name}...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize services here (vector store, database connections, etc.)
    # TODO: Initialize FAISS index
    # TODO: Initialize Supabase client
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    # Cleanup resources here


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
            "database": "pending",  # TODO: Check Supabase connection
            "vector_store": "pending",  # TODO: Check FAISS status
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
