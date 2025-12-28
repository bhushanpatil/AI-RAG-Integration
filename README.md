RAG Learning Project
A comprehensive Retrieval-Augmented Generation (RAG) system for learning purposes, supporting multiple data sources including files, databases, and web content.
🎯 Features

Multiple Data Sources:

Text files (.txt, .md)
PDF documents
DOCX files
PowerPoint presentations
CSV files
MySQL databases
PostgreSQL databases
Wikipedia articles
Web pages and custom web scraping


Vector Storage: ChromaDB for efficient similarity search
Embeddings: Open-source sentence transformers
Modular Architecture: Easy to extend and customize
RAG Pipeline: Complete retrieval and question-answering system

📋 Prerequisites

Python 3.11
pip (Python package manager)
Optional: MySQL/PostgreSQL for database examples

🚀 Quick Start
1. Clone and Setup
bash# Create project directory
mkdir rag-learning-project
cd rag-learning-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
2. Install Dependencies
bashpip install -r requirements.txt
3. Configure Environment
bash# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional for databases)