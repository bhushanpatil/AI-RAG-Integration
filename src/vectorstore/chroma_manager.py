
from typing import Any, Dict, List, Optional
from langchain.embeddings.base import Embeddings
from pathlib import Path
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class ChromaManager:
    """ Manage ChromaDB Vector store operations """
    
    def __init__(self, embedding_function: Embeddings, 
                 persist_dir_path: str, 
                 collection_name: str = "rag_collection",
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """
        Initialize ChromaDB manager
        
        Args:
            embedding_function: LangChain embeddings object
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_function = embedding_function
        self.persist_dir_path = Path(persist_dir_path)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        #create directory if doens't exist 
        self.persist_dir_path.mkdir(parents=True,exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
            )
        
        # Initialize vector store
        self.vectorstore = None
        self._init_vectorstore()
        
    def _init_vectorstore(self):
        """ Initialise or load existing vector store """
        try:  
            self.vectorstore = Chroma(
                collection_name = self.collection_name,
                embedding_function = self.embedding_function,
                persist_directory = str(self.persist_dir_path),
            )
            print(f"Loaded existing chromadb collection {self.collection_name}")
        except Exception as e:
            print(f"Creating new chromadb collection {self.collection_name}")
            self.vectorstore = Chroma(
                collection_name = self.collection_name,
                embedding_function = self.embedding_function,
                persist_directory = str(self.persist_dir_path),
            )
    
    def add_documents(self, documents: List[Document]):
        """ Add documents to chroma db vector store """

        split_docs = self.text_splitter.split_documents(documents=documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        
        #add vector to store
        doc_ids = self.vectorstore.add_documents(split_docs)
        
        #vector store persist
        self.vectorstore.persist()
        return doc_ids   
    
        
        
    def similarity_search(self, query:str, 
                          k:int = 5, 
                          filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        if filter:
            results = self.vectorstore.similarity_search(query,k,filter)
        else:
            results = self.vectorstore.similarity_search(query,k)
        
        return results

    def persist(self):
        """Manually persist the vector store"""
        self.vectorstore.persist()
        print("Vector store persisted successfully")
        
    def as_retriever(self, search_type: str = "similarity", k:int = 5):
        return self.vectorstore.as_retriever(search_type=search_type,search_kwargs={
            "k": k
        })