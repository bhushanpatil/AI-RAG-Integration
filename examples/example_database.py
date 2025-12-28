"""
Example: Loading data from databases
Note: Requires MySQL or PostgreSQL to be installed and configured
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import settings
from src.data_sources.database_sources import MySQLSource, PostgreSQLSource
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.chroma_manager import ChromaManager


def example_mysql():
    """Example: Load data from MySQL"""
    print("\n" + "="*50)
    print("MySQL Database Example")
    print("="*50)
    
    # Check if MySQL credentials are configured
    if not all([settings.MYSQL_HOST, settings.MYSQL_USER, settings.MYSQL_PASSWORD]):
        print("!!!  MySQL credentials not configured in .env file")
        print("Please set: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")
        return None
    
    try:
        # Initialize MySQL source
        mysql_source = MySQLSource(
            host=settings.MYSQL_HOST,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            port=settings.MYSQL_PORT
        )
        
        # Validate connection
        if not mysql_source.validate_source():
            print(" Could not connect to MySQL database")
            return None
        
        print(" Connected to MySQL database")
        
        # List available tables
        tables = mysql_source.list_tables()
        print(f"\nAvailable tables: {tables}")
        
        # Load data from a table
        if tables:
            table_name = tables[0]
            print(f"\nLoading data from table: {table_name}")
            
            # Get table info
            table_info = mysql_source.get_table_info(table_name)
            print(f"Table columns: {[col['COLUMN_NAME'] for col in table_info]}")
            
            # Load documents
            documents = mysql_source.load(table_name=table_name)
            print(f"Loaded {len(documents)} documents")
            
            # Show sample
            if documents:
                print(f"\nSample document:")
                print(f"Content: {documents[0].page_content[:200]}...")
                print(f"Metadata: {documents[0].metadata}")
            
            return documents
        else:
            print("No tables found in database")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


def example_postgresql():
    """Example: Load data from PostgreSQL"""
    print("\n" + "="*50)
    print("PostgreSQL Database Example")
    print("="*50)
    
    # Check if PostgreSQL credentials are configured
    if not all([settings.POSTGRES_HOST, settings.POSTGRES_USER, settings.POSTGRES_PASSWORD]):
        print("!!!  PostgreSQL credentials not configured in .env file")
        print("Please set: POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE")
        return None
    
    try:
        # Initialize PostgreSQL source
        postgres_source = PostgreSQLSource(
            host=settings.POSTGRES_HOST,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DATABASE,
            port=settings.POSTGRES_PORT
        )
        
        # Validate connection
        if not postgres_source.validate_source():
            print(" Could not connect to PostgreSQL database")
            return None
        
        print(" Connected to PostgreSQL database")
        
        # List available tables
        tables = postgres_source.list_tables()
        print(f"\nAvailable tables: {tables}")
        
        # Load data from a table
        if tables:
            table_name = tables[0]
            print(f"\nLoading data from table: {table_name}")
            
            # Get table info
            table_info = postgres_source.get_table_info(table_name)
            print(f"Table columns: {[col['column_name'] for col in table_info]}")
            
            # Load documents
            documents = postgres_source.load(table_name=table_name)
            print(f"Loaded {len(documents)} documents")
            
            # Show sample
            if documents:
                print(f"\nSample document:")
                print(f"Content: {documents[0].page_content[:200]}...")
                print(f"Metadata: {documents[0].metadata}")
            
            return documents
        else:
            print("No tables found in database")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


def example_custom_query():
    """Example: Load data using custom SQL query"""
    print("\n" + "="*50)
    print("Custom SQL Query Example")
    print("="*50)
    
    # This example uses MySQL, but works similarly for PostgreSQL
    if not all([settings.MYSQL_HOST, settings.MYSQL_USER, settings.MYSQL_PASSWORD]):
        print("!!!  MySQL credentials not configured")
        return None
    
    try:
        mysql_source = MySQLSource(
            host=settings.MYSQL_HOST,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            port=settings.MYSQL_PORT
        )
        
        if not mysql_source.validate_source():
            return None
        
        # Custom query example
        # Adjust this query based on your actual database schema
        query = """
        SELECT id, name, description 
        FROM products 
        WHERE category = 'electronics' 
        LIMIT 10
        """
        
        print(f"Executing query:\n{query}")
        
        # Load with specific columns for content and metadata
        documents = mysql_source.load(
            query=query,
            content_columns=['name', 'description'],
            metadata_columns=['id', 'category']
        )
        
        print(f"Loaded {len(documents)} documents from query")
        
        return documents
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def integrate_with_vectorstore(documents):
    """Integrate database documents with vector store"""
    if not documents:
        print("\nNo documents to add to vector store")
        return
    
    print("\n" + "="*50)
    print("Adding to Vector Store")
    print("="*50)
    
    # Initialize embeddings
    embedding_manager = EmbeddingManager(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # Initialize ChromaDB
    chroma_manager = ChromaManager(
        embedding_function=embedding_manager.get_embeddings(),
        persist_directory=str(settings.CHROMA_DIR),
        collection_name="database_docs"
    )
    
    # Add documents
    doc_ids = chroma_manager.add_documents(documents)
    print(f"Added {len(doc_ids)} document chunks to vector store")
    
    # Test search
    print("\nTesting similarity search...")
    query = "product information"
    results = chroma_manager.similarity_search(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} relevant documents:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:150]}...")
        print(f"   Source: {doc.metadata.get('database', 'unknown')}\n")


def main():
    """Main function"""
    print("="*50)
    print("Database Sources Example")
    print("="*50)
    print("\nThis example demonstrates loading data from databases.")
    print("Make sure to configure database credentials in .env file\n")
    
    # Try MySQL
    mysql_docs = example_mysql()
    
    # Try PostgreSQL
    postgres_docs = example_postgresql()
    
    # Try custom query
    # custom_docs = example_custom_query()
    
    # Integrate with vector store if we have documents
    all_docs = []
    if mysql_docs:
        all_docs.extend(mysql_docs)
    if postgres_docs:
        all_docs.extend(postgres_docs)
    
    if all_docs:
        integrate_with_vectorstore(all_docs)
    else:
        print("\n" + "="*50)
        print("Database Configuration Guide")
        print("="*50)
        print("""
To use database sources:

1. Install database server (MySQL or PostgreSQL)

2. Create a test database and table:
   
   MySQL:
   CREATE DATABASE test_db;
   USE test_db;
   CREATE TABLE products (
       id INT PRIMARY KEY,
       name VARCHAR(255),
       description TEXT,
       category VARCHAR(100)
   );
   INSERT INTO products VALUES 
   (1, 'Laptop', 'High-performance laptop', 'electronics'),
   (2, 'Phone', 'Smartphone with AI features', 'electronics');

3. Update .env file with credentials:
   MYSQL_HOST=localhost
   MYSQL_USER=your_user
   MYSQL_PASSWORD=your_password
   MYSQL_DATABASE=test_db

4. Run this example again
        """)


if __name__ == "__main__":
    main()