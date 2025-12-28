"""
Example: Loading data from web sources
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import settings
from src.data_sources.web_sources import (
    WikipediaSource,
    WebPageSource,
    MultiWebPageSource
)
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.chroma_manager import ChromaManager
from src.rag.qa_chain import SimpleRAGChain


def example_wikipedia():
    """Load data from Wikipedia"""
    print("\n" + "="*50)
    print("Wikipedia Source Example")
    print("="*50)
    
    try:
        # Search for topics
        wiki_source = WikipediaSource("Artificial Intelligence")
        
        print(f"\nSearching Wikipedia for: Artificial Intelligence")
        topics = wiki_source.search_topics(max_results=5)
        print(f"Related topics: {topics}")
        
        # Load articles
        print(f"\nLoading Wikipedia articles...")
        documents = wiki_source.load(max_docs=2)
        
        print(f"Loaded {len(documents)} documents")
        
        # Show information
        for i, doc in enumerate(documents, 1):
            print(f"\nDocument {i}:")
            print(f"  Title: {doc.metadata.get('title', 'N/A')}")
            print(f"  Summary: {doc.metadata.get('summary', 'N/A')[:200]}...")
            print(f"  Content length: {len(doc.page_content)} characters")
        
        return documents
        
    except Exception as e:
        print(f"Error loading from Wikipedia: {e}")
        return []


def example_web_page():
    """Load data from a web page"""
    print("\n" + "="*50)
    print("Web Page Source Example")
    print("="*50)
    
    # Example: Load Python.org homepage
    url = "https://www.python.org/about/"
    
    try:
        web_source = WebPageSource(url)
        
        print(f"\nLoading web page: {url}")
        
        if not web_source.validate_source():
            print("Cannot access URL")
            return []
        
        documents = web_source.load()
        print(f"Loaded {len(documents)} document(s)")
        
        if documents:
            print(f"\nContent preview:")
            print(documents[0].page_content[:300] + "...")
            print(f"\nMetadata: {documents[0].metadata}")
        
        return documents
        
    except Exception as e:
        print(f"Error loading web page: {e}")
        return []


def example_multiple_pages():
    """Load data from multiple web pages"""
    print("\n" + "="*50)
    print("Multiple Web Pages Example")
    print("="*50)
    
    urls = [
        "https://www.python.org/about/",
        "https://www.python.org/about/apps/",
    ]
    
    try:
        multi_source = MultiWebPageSource(urls)
        
        print(f"\nLoading {len(urls)} web pages...")
        documents = multi_source.load()
        
        print(f"Loaded {len(documents)} documents")
        
        for i, doc in enumerate(documents, 1):
            print(f"\nPage {i}:")
            print(f"URL: {doc.metadata.get('source', 'N/A')}")
            print(f"Content length: {len(doc.page_content)} characters")
            print(f"Preview: {doc.page_content[:150]}...")
        
        return documents
        
    except Exception as e:
        print(f"Error loading web pages: {e}")
        return []


def integrate_with_rag(documents):
    """Integrate web documents with RAG system"""
    if not documents:
        print("\nNo documents to add to RAG system")
        return
    
    print("\n" + "="*50)
    print("Building RAG System with Web Data")
    print("="*50)
    
    # Initialize embeddings
    print("\n1. Initializing embeddings...")
    embedding_manager = EmbeddingManager(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # Initialize vector store
    print("2. Setting up vector store...")
    chroma_manager = ChromaManager(
        embedding_function=embedding_manager.get_embeddings(),
        persist_dir_path=str(settings.CHROMA_DIR),
        collection_name="web_docs"
    )
    
    # Clear existing collection
    chroma_manager.delete_collection()
    
    # Add documents
    print("3. Adding documents to vector store...")
    doc_ids = chroma_manager.add_documents(documents)
    print(f"   Added {len(doc_ids)} document chunks")
    
    # Create RAG chain
    print("4. Creating RAG chain...")
    rag_chain = SimpleRAGChain(chroma_manager)
    
    # Run sample queries
    print("\n" + "="*50)
    print("Testing RAG Queries")
    print("="*50)
    
    queries = [
        "What is Python used for?",
        "Tell me about artificial intelligence",
        "What are the applications of Python?"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = rag_chain.get_answer_with_context(query, k=3)
        
        print(f"\nRetrieved {result['num_documents']} relevant documents:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. Source: {source['source']}")
            print(f"   Type: {source['source_type']}")
            print(f"   Preview: {source['content_preview'][:150]}...")


def main():
    """Main function"""
    print("="*50)
    print("Web Sources Example")
    print("="*50)
    
    all_documents = []
    
    # Example 1: Wikipedia
    wiki_docs = example_wikipedia()
    all_documents.extend(wiki_docs)
    
    # Example 2: Single web page
    web_docs = example_web_page()
    all_documents.extend(web_docs)
    
    # Example 3: Multiple pages
    multi_docs = example_multiple_pages()
    all_documents.extend(multi_docs)
    
    # Example 4: Custom scraper
    # custom_docs = example_custom_scraper()
    # all_documents.extend(custom_docs)
    
    # Example 5: Specific content
    # specific_docs = example_specific_content()
    # all_documents.extend(specific_docs)
    
    # Integrate with RAG
    if all_documents:
        print(f"\n{'='*50}")
        print(f"Total documents collected: {len(all_documents)}")
        integrate_with_rag(all_documents)
    else:
        print("\n No documents were loaded")
        print("\nTroubleshooting:")
        print("- Check internet connection")
        print("- Some websites may block scraping")
        print("- Try different URLs or Wikipedia topics")


if __name__ == "__main__":
    main()