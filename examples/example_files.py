"""
Example: Loading data from various file formats
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import settings
from src.data_sources.file_sources import TextFileSource
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.chroma_manager import ChromaManager


def example_text_file():
    """Load a text file"""
    print("\n" + "="*50)
    print("Loading Text File")
    print("="*50)
    
    # Create sample text file
    sample_file = settings.DATA_DIR / "documents" / "sample.txt"
    # sample_file.write_text("""
    # Artificial Intelligence (AI) is transforming the world.
    # Machine learning is a subset of AI that enables computers to learn from data.
    # Deep learning uses neural networks with multiple layers.
    # Natural Language Processing (NLP) helps computers understand human language.
    # """)
    
    # Load the file
    source = TextFileSource(sample_file)
    documents = source.load()
    
    print(f"Loaded {len(documents)} document(s)")
    print(f"Content preview: {documents[0].page_content[:200]}...")
    print(f"Metadata: {documents[0].metadata}")
    
    return documents


def main():
    """Main example function"""
    print("File Sources Example")
    print("="*50)
    
    # Load different file types
    text_docs = example_text_file()
        
    # Initialize embedding manager
    print("\n" + "="*50)
    print("Initializing Embeddings and Vector Store")
    print("="*50)
    
    embedding_manager = EmbeddingManager(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # Initialize ChromaDB
    chroma_manager = ChromaManager(
        embedding_function=embedding_manager.get_embeddings(),
        persist_dir_path=str(settings.CHROMA_DIR),
        collection_name="file_examples"
    )
    
    # Add documents to vector store
    print("\nAdding documents to vector store...")
    doc_ids = chroma_manager.add_documents(text_docs)
    print(f"Added {len(doc_ids)} document chunks to ChromaDB")
    
    
    # Test similarity search
    print("\n" + "="*50)
    print("Testing Similarity Search")
    print("="*50)
    
    query = "What is machine learning?"
    results = chroma_manager.similarity_search(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} relevant documents:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. Source: {doc.metadata.get('source_type', 'unknown')}")
        print(f"   Content: {doc.page_content[:150]}...")
        print()


if __name__ == "__main__":
    main()