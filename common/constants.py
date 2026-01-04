"""
This module defines the constants for RAG application.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Openrouter credentials
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Vector store
KNOWLEDGE_BASE_DIR_PATH = "knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "vector_db"
VECTOR_DB_COLLECTION_NAME = "msgm_annual_reports"
CHROMA_DB_COLLECTION_MAX_BATCH_SIZE = 5461
EMBEDDINGS_RETRIEVAL_COUNT = 10

# LLM interaction models
CHUNKS_RERANKING_MODEL = "openai/gpt-oss-120b:free"
CHAT_MODEL = "openai/gpt-oss-120b:free"
