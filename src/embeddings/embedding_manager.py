"""
Embedding model manager - using open source models
"""

from typing import List
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingManager:
    """Manager for embedding models"""
    
    AVAILABLE_MODELS = {
         "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",  # Fast, good for most tasks
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",  # Better quality
        "multi-qa-MiniLM-L6-cos-v1": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # Optimized for Q&A
        "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",  # Balance speed/quality
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embedding: bool = True
        ):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of the model (from AVAILABLE_MODELS)
            device: Device to run on ('cpu' or 'cuda')
            normalize_embeddings: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embedding
        self.embeddings = self._load_embeddings()
        
    def _load_embeddings(self) -> Embeddings:
        """Load the embedding model"""
        if self.model_name in self.AVAILABLE_MODELS:
            model_path = self.AVAILABLE_MODELS[self.model_name]
        else:
            # Assume it's a full model path
            model_path = self.model_name
        
        print(f"Loading embedding model: {model_path}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': self.device},
            encode_kwargs={
                'normalize_embeddings': self.normalize_embeddings,
                'batch_size': 32
            }
        )
        
        print("Embedding model loaded successfully")
        return embeddings
    
    def get_embeddings(self) -> Embeddings:
        return self.embeddings
    
    def embed_text(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)