"""
Configuration settings for the RAG system
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda" for GPU
    
    # ChromaDB settings
    CHROMA_COLLECTION_NAME: str = "rag_documents"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Database settings (optional - load from .env)
    MYSQL_HOST: Optional[str] = None
    MYSQL_PORT: int = 3306
    MYSQL_USER: Optional[str] = None
    MYSQL_PASSWORD: Optional[str] = None
    MYSQL_DATABASE: Optional[str] = None
    
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DATABASE: Optional[str] = None
    
    # Web scraping settings
    USER_AGENT: str = "RAG-Learning-Bot/1.0"
    REQUEST_TIMEOUT: int = 30
    
    # RAG settings
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.5
    
    #API keys
    GOOGLE_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Create necessary directories
settings.DATA_DIR.mkdir(exist_ok=True)
settings.CHROMA_DIR.mkdir(exist_ok=True)
(settings.DATA_DIR / "documents").mkdir(exist_ok=True)
(settings.DATA_DIR / "databases").mkdir(exist_ok=True)
(settings.DATA_DIR / "logs").mkdir(exist_ok=True)