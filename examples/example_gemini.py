

import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent.parent))

from src.config import settings
from src.data_sources.file_sources import TextFileSource
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.chroma_manager import ChromaManager
from src.rag.qa_chain import SimpleRAGChain
import google.generativeai as genai


# Get Google API key
API_KEY = os.getenv("GOOGLE_API_KEY", settings.GOOGLE_API_KEY)
genai.configure(api_key=API_KEY)
gemini = genai.GenerativeModel('gemini-2.5-flash-lite')

doc_path = settings.DATA_DIR / "documents" / "sample.txt"

print("\nLoading document into ChromaDB...")

source = TextFileSource(doc_path)
documents = source.load()
print(f"Loaded {len(documents)} document")

embeddings = EmbeddingManager(model_name="all-MiniLM-L6-v2").get_embeddings()

# Store in ChromaDB
chroma = ChromaManager(
    embedding_function=embeddings,
    persist_dir_path=str(settings.CHROMA_DIR),
    collection_name="sample_text_demo"
)
#@todo - delete old data
chroma.add_documents(documents)

print("Documents added to vector store")

# Initialize RAG Chain
rag_chain = SimpleRAGChain(chroma)

def ask_question (question: str):
    print("\n Using RAG chain to retrieve context...")
    result = rag_chain.get_answer_with_context(question, k=2)
    
    print(f"   Found {result['num_documents']} relevant chunks")
    for i, source in enumerate(result['sources'], 1):
        print(f"   Chunk {i}: {source['content_preview'][:80]}...")
    
    # 2. Create prompt using RAG chain's template
    print("\n Creating prompt with context...")
    prompt = rag_chain.create_prompt(question, result['context'])
    
    # 3. Get answer from Gemini
    print("\n Asking Gemini...")
    response = gemini.generate_content(prompt)
    
    # 4. Show answer
    print("\n Answer:")
    print("-" * 60)
    print(response.text)
    print("-" * 60)
    
    return response.text


if __name__ == "__main__":
     ask_question("What is Machine learning used for?")