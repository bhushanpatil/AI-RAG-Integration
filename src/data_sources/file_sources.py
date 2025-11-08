"""
File-based data sources: Text, PDF, DOCX, PPT, CSV
"""

from typing import List, Union
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from .base import BaseDataSource

class TextFileSource(BaseDataSource):
    """ Load text files """
    def __init__(self, file_path: Union[str, Path]):
        super().__init__(source_name=f"Textfile: {file_path}")
        self.file_path = Path(file_path)
    
    def validate_source(self) -> bool:
        return self.file_path.exists() and self.file_path.suffix in ['.txt', '.md']
    
    def load(self, encoding: str = "utf-8") -> List[Document]:
        """Load text file"""
        
        if not self.validate_source():
            raise ValueError(f"Invalid file path {self.file_path}")
        
        loader = TextLoader(str(self.file_path), encoding=encoding)
        self.documents = loader.load()
        
        #add metadata
        for docs in self.documents:
            docs.metadata.update({
                "source_type": "text",
                "file_name": self.file_path.name 
                "file_path": str(self.file_path)
            })
            
        return self.documents