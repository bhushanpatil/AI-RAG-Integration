"""
Database data sources: MySQL, PostgreSQL
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text
import pandas as pd
from .base import BaseDataSource


class MySQLSource(BaseDataSource):
    """Load data from MySQL database"""
    
    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306
    ):
        super().__init__(source_name=f"MySQL: {database}")
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection_string = (
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        )
        self.engine = None
    
    def validate_source(self) -> bool:
        """Test database connection"""
        try:
            self.engine = create_engine(self.connection_string)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"MySQL connection error: {e}")
            return False
    
    def load(
        self,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load data from MySQL
        
        Args:
            query: Custom SQL query
            table_name: Table name to load entire table
            content_columns: Columns to use as document content
            metadata_columns: Columns to use as metadata
        """
        if not self.validate_source():
            raise ValueError("Cannot connect to MySQL database")
        
        # Execute query or load table
        if query:
            df = pd.read_sql(query, self.engine)
        elif table_name:
            df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
        else:
            raise ValueError("Either query or table_name must be provided")
        
        # Convert to documents
        self.documents = []
        
        for idx, row in df.iterrows():
            # Determine content
            if content_columns:
                content = " | ".join(
                    f"{col}: {row[col]}" for col in content_columns if col in df.columns
                )
            else:
                content = " | ".join(f"{col}: {val}" for col, val in row.items())
            
            # Determine metadata
            metadata = {
                'source_type': 'mysql',
                'database': self.database,
                'row_index': idx
            }
            
            if metadata_columns:
                for col in metadata_columns:
                    if col in df.columns:
                        metadata[col] = row[col]
            
            doc = Document(page_content=content, metadata=metadata)
            self.documents.append(doc)
        
        return self.documents
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        if not self.engine:
            self.validate_source()
        
        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{self.database}' AND TABLE_NAME = '{table_name}'
        """
        df = pd.read_sql(query, self.engine)
        return df.to_dict('records')
    
    def list_tables(self) -> List[str]:
        """List all tables in the database"""
        if not self.engine:
            self.validate_source()
        
        query = f"SHOW TABLES FROM {self.database}"
        df = pd.read_sql(query, self.engine)
        return df.iloc[:, 0].tolist()


class PostgreSQLSource(BaseDataSource):
    """Load data from PostgreSQL database"""
    
    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 5432
    ):
        super().__init__(source_name=f"PostgreSQL: {database}")
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection_string = (
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        )
        self.engine = None
    
    def validate_source(self) -> bool:
        """Test database connection"""
        try:
            self.engine = create_engine(self.connection_string)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"PostgreSQL connection error: {e}")
            return False
    
    def load(
        self,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load data from PostgreSQL
        
        Args:
            query: Custom SQL query
            table_name: Table name to load entire table
            content_columns: Columns to use as document content
            metadata_columns: Columns to use as metadata
        """
        if not self.validate_source():
            raise ValueError("Cannot connect to PostgreSQL database")
        
        # Execute query or load table
        if query:
            df = pd.read_sql(query, self.engine)
        elif table_name:
            df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
        else:
            raise ValueError("Either query or table_name must be provided")
        
        # Convert to documents
        self.documents = []
        
        for idx, row in df.iterrows():
            # Determine content
            if content_columns:
                content = " | ".join(
                    f"{col}: {row[col]}" for col in content_columns if col in df.columns
                )
            else:
                content = " | ".join(f"{col}: {val}" for col, val in row.items())
            
            # Determine metadata
            metadata = {
                'source_type': 'postgresql',
                'database': self.database,
                'row_index': idx
            }
            
            if metadata_columns:
                for col in metadata_columns:
                    if col in df.columns:
                        metadata[col] = row[col]
            
            doc = Document(page_content=content, metadata=metadata)
            self.documents.append(doc)
        
        return self.documents
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        if not self.engine:
            self.validate_source()
        
        query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        """
        df = pd.read_sql(query, self.engine)
        return df.to_dict('records')
    
    def list_tables(self) -> List[str]:
        """List all tables in the database"""
        if not self.engine:
            self.validate_source()
        
        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        """
        df = pd.read_sql(query, self.engine)
        return df['table_name'].tolist()