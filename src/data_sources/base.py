
"""
Base class for all data sources
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.schema import Document
class BaseDataSource(ABC):
    """Abstract base class for all data sources"""
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.documents: List[Document] = []
    
    @abstractmethod
    def load(self, **kwargs) -> List[Document]:
        """
        Load data from the source and return as Document objects
        
        Returns:
            List[Document]: List of LangChain Document objects
        """
        pass
    
    @abstractmethod
    def validate_source(self) -> bool:
        """
        Validate that the data source is accessible
        
        Returns:
            bool: True if source is valid, False otherwise
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
         Get metadata about the loaded documents
        
        Returns:
            Dict containing document count and source info
        """
        return {
            "source_name": self.source_name,
            "document_count": len(self.documents),
            "total_charachters": sum(len(doc.page_content) for doc in self.documents)
        }
   
    def clear(self):
        """Clear loaded documents"""
        self.documents = []
    
    def __repr__(self):
        return f"{self.__class__.__name__}(Sourcename: {self.source_name}, Documents: {len(self.documents)})"