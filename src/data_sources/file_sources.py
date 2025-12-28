"""
File-based data sources: Text, PDF, DOCX, PPT, CSV
"""

from typing import List, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
)
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
                "file_name": self.file_path.name,
                "file_path": str(self.file_path)
            })
            
        return self.documents
    
class PDFSource(BaseDataSource):
    """Load PDF files"""
    
    def __init__(self, file_path: Union[str, Path]):
        super().__init__(source_name=f"PDF: {file_path}")
        self.file_path = Path(file_path)
    
    def validate_source(self) -> bool:
        return self.file_path.exists() and self.file_path.suffix == '.pdf'
    
    def load(self) -> List[Document]:
        """Load PDF file"""
        if not self.validate_source():
            raise ValueError(f"Invalid PDF file: {self.file_path}")
        
        loader = PyPDFLoader(str(self.file_path))
        self.documents = loader.load()
        
        # Add metadata
        for doc in self.documents:
            doc.metadata.update({
                'source_type': 'pdf',
                'file_name': self.file_path.name,
                'file_path': str(self.file_path)
            })
        
        return self.documents


class DocxSource(BaseDataSource):
    """Load DOCX files"""
    
    def __init__(self, file_path: Union[str, Path]):
        super().__init__(source_name=f"DOCX: {file_path}")
        self.file_path = Path(file_path)
    
    def validate_source(self) -> bool:
        return self.file_path.exists() and self.file_path.suffix == '.docx'
    
    def load(self) -> List[Document]:
        """Load DOCX file"""
        if not self.validate_source():
            raise ValueError(f"Invalid DOCX file: {self.file_path}")
        
        loader = Docx2txtLoader(str(self.file_path))
        self.documents = loader.load()
        
        # Add metadata
        for doc in self.documents:
            doc.metadata.update({
                'source_type': 'docx',
                'file_name': self.file_path.name,
                'file_path': str(self.file_path)
            })
        
        return self.documents


class PowerPointSource(BaseDataSource):
    """Load PowerPoint files"""
    
    def __init__(self, file_path: Union[str, Path]):
        super().__init__(source_name=f"PPT: {file_path}")
        self.file_path = Path(file_path)
    
    def validate_source(self) -> bool:
        return self.file_path.exists() and self.file_path.suffix in ['.ppt', '.pptx']
    
    def load(self) -> List[Document]:
        """Load PowerPoint file"""
        if not self.validate_source():
            raise ValueError(f"Invalid PowerPoint file: {self.file_path}")
        
        loader = UnstructuredPowerPointLoader(str(self.file_path))
        self.documents = loader.load()
        
        # Add metadata
        for doc in self.documents:
            doc.metadata.update({
                'source_type': 'powerpoint',
                'file_name': self.file_path.name,
                'file_path': str(self.file_path)
            })
        
        return self.documents


class CSVSource(BaseDataSource):
    """Load CSV files"""
    
    def __init__(self, file_path: Union[str, Path]):
        super().__init__(source_name=f"CSV: {file_path}")
        self.file_path = Path(file_path)
    
    def validate_source(self) -> bool:
        return self.file_path.exists() and self.file_path.suffix == '.csv'
    
    def load(self, source_column: str = None) -> List[Document]:
        """
        Load CSV file
        
        Args:
            source_column: Specific column to use as content (optional)
        """
        if not self.validate_source():
            raise ValueError(f"Invalid CSV file: {self.file_path}")
        
        if source_column:
            loader = CSVLoader(str(self.file_path), source_column=source_column)
        else:
            loader = CSVLoader(str(self.file_path))
        
        self.documents = loader.load()
        
        # Add metadata
        for doc in self.documents:
            doc.metadata.update({
                'source_type': 'csv',
                'file_name': self.file_path.name,
                'file_path': str(self.file_path)
            })
        
        return self.documents


class DirectorySource(BaseDataSource):
    """Load all supported files from a directory"""
    
    def __init__(self, directory_path: Union[str, Path]):
        super().__init__(source_name=f"Directory: {directory_path}")
        self.directory_path = Path(directory_path)
        self.supported_extensions = {
            '.txt': TextFileSource,
            '.md': TextFileSource,
            '.pdf': PDFSource,
            '.docx': DocxSource,
            '.pptx': PowerPointSource,
            '.csv': CSVSource,
        }
    
    def validate_source(self) -> bool:
        return self.directory_path.exists() and self.directory_path.is_dir()
    
    def load(self, recursive: bool = False) -> List[Document]:
        """
        Load all supported files from directory
        
        Args:
            recursive: Whether to search subdirectories
        """
        if not self.validate_source():
            raise ValueError(f"Invalid directory: {self.directory_path}")
        
        self.documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in self.directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                try:
                    source_class = self.supported_extensions[file_path.suffix]
                    source = source_class(file_path)
                    docs = source.load()
                    self.documents.extend(docs)
                    print(f"Loaded {len(docs)} documents from {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
        
        return self.documents