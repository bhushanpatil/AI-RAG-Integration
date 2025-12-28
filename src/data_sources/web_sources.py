"""
Web-based data sources: Wikipedia, websites
"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WikipediaLoader,
    WebBaseLoader,
)
import wikipedia
import requests
from bs4 import BeautifulSoup
from .base import BaseDataSource


class WikipediaSource(BaseDataSource):
    """Load data from Wikipedia"""
    
    def __init__(self, query: str, lang: str = "en"):
        super().__init__(source_name=f"Wikipedia: {query}")
        self.query = query
        self.lang = lang
        wikipedia.set_lang(lang)
    
    def validate_source(self) -> bool:
        """Check if Wikipedia query returns results"""
        try:
            results = wikipedia.search(self.query, results=1)
            return len(results) > 0
        except Exception as e:
            print(f"Wikipedia validation error: {e}")
            return False
    
    def load(self, max_docs: int = 2, load_all_available_meta: bool = True) -> List[Document]:
        """
        Load Wikipedia articles
        
        Args:
            max_docs: Maximum number of documents to load
            load_all_available_meta: Load all available metadata
        """
        if not self.validate_source():
            raise ValueError(f"No Wikipedia results for query: {self.query}")
        
        loader = WikipediaLoader(
            query=self.query,
            lang=self.lang,
            load_max_docs=max_docs,
            load_all_available_meta=load_all_available_meta
        )
        self.documents = loader.load()
        
        # Add source type to metadata
        for doc in self.documents:
            doc.metadata['source_type'] = 'wikipedia'
            doc.metadata['language'] = self.lang
            doc.metadata['categories'] = str(doc.categories)
        
        return self.documents
    
    def search_topics(self, max_results: int = 10) -> List[str]:
        """Search for related Wikipedia topics"""
        return wikipedia.search(self.query, results=max_results)


class WebPageSource(BaseDataSource):
    """Load data from web pages"""
    
    def __init__(self, url: str):
        super().__init__(source_name=f"WebPage: {url}")
        self.url = url
    
    def validate_source(self) -> bool:
        """Check if URL is accessible"""
        try:
            response = requests.head(self.url, timeout=10, allow_redirects=True)
            return response.status_code == 200
        except Exception as e:
            print(f"URL validation error: {e}")
            return False
    
    def load(self) -> List[Document]:
        """Load web page content"""
        if not self.validate_source():
            raise ValueError(f"Cannot access URL: {self.url}")
        
        loader = WebBaseLoader(self.url)
        self.documents = loader.load()
        
        # Add metadata
        for doc in self.documents:
            doc.metadata.update({
                'source_type': 'webpage',
                'url': self.url,
                'categories': str(doc.categories)
            })
        
        return self.documents


class MultiWebPageSource(BaseDataSource):
    """Load data from multiple web pages"""
    
    def __init__(self, urls: List[str]):
        super().__init__(source_name=f"MultiWebPage: {len(urls)} URLs")
        self.urls = urls
    
    def validate_source(self) -> bool:
        """Check if at least one URL is accessible"""
        for url in self.urls:
            try:
                response = requests.head(url, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    return True
            except:
                continue
        return False
    
    def load(self) -> List[Document]:
        """Load multiple web pages"""
        loader = WebBaseLoader(self.urls)
        self.documents = loader.load()
        
        # Add metadata
        for doc in self.documents:
            doc.metadata['source_type'] = 'webpage'
        
        return self.documents
