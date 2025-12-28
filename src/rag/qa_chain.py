
"""
RAG Question-Answering Chain
This module provides a complete RAG pipeline for question answering
"""

from langchain_core.documents import Document
from typing import Dict, List
class SimpleRAGChain:
    """
    Simple RAG chain without external LLM
    Useful for learning and testing retrieval
    """
    
    def __init__(self, vectorstore_manager):
        """
        """
        self.vectorstore_manager = vectorstore_manager
        self.retriever = vectorstore_manager.as_retriever()
        
    def retrieve(self, query:str, k:int = 5, search_type:str = "similarity") -> List[Document]:
        self.retriever = self.vectorstore_manager.as_retriever(
            search_type= search_type,
            k=k
        )
        return self.retriever.invoke(query)
    
    def get_prompt_template(self) -> str:
        """
        Get a template for question answering with context
        This can be used with external LLMs
        """
        template = """You are a helpful AI assistant. Use the following context to answer the question.
                If you cannot find the answer in the context, say "I cannot find the answer in the provided context."

                Context:
                {context}

                Question: {question}

                Answer:"""
        return template
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a formatted prompt for LLM
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Formatted prompt
        """
        template = self.get_prompt_template()
        return template.format(context=context, question=query)
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        """
        Retrieve documents with relevance scores
        
        Args:
            query: User query
            k: Number of documents
        
        Returns:
            List of (Document, score) tuples
        """
        return self.vectorstore_manager.similarity_search_with_score(query, k=k)
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: Retrieved documents
        
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            context_parts.append(f"[Document {i} - Source: {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def get_answer_with_context(self, query: str, k: int=5, search_type: str = "similarity", score_threshold: float = 0.0) -> Dict[str, any]:
        """
        Get answer with retrived context
        """
        if search_type == "similarity_with_score":
            docs_with_scores = self.retrieve_with_score(query, k)
            # Filter by score threshold
            filtered = [(doc, score) for doc, score in docs_with_scores if score > score_threshold]
            documents = [doc for doc, score in filtered]
            scores = [score for doc,score in filtered]
        else:
            documents = self.retrieve(query,k=k, search_type=search_type)
            scores = None

        # Format context
        context = self.format_context(documents)
        
        # Extract sources
        sources = []
        for doc in documents:
            source_info = {
                'source_type': doc.metadata.get('source_type', 'unknown'),
                'source': doc.metadata.get('source', doc.metadata.get('file_name', 'Unknown')),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        result = {
            'query': query,
            'context': context,
            'documents': documents,
            'sources': sources,
            'num_documents': len(documents)
        }
        
        if scores:
            result['scores'] = scores
        
        return result
