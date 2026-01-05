"""
This module defines the logic for reading and ingesting documents for RAG application.
"""

import os
import glob
from argparse import ArgumentParser
from chromadb import PersistentClient
from chromadb.api import ClientAPI
from pypdf import PdfReader
from typing import Any, Generator, List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from common.models import Chunk
from common.constants import (
    KNOWLEDGE_BASE_DIR_PATH,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    VECTOR_DB_COLLECTION_NAME,
    CHROMA_DB_COLLECTION_MAX_BATCH_SIZE,
)


##### Helper functions #####
def fetch_documents(knowledge_base_dir_path: str) -> List[Document]:
    pdf_documents_pattern = f"{knowledge_base_dir_path}/**/*.pdf"
    document_paths = glob.glob(pdf_documents_pattern, recursive=True)
    documents = []
    for document_path in document_paths:
        name = document_path.split("/")[-1]
        pdf_reader = PdfReader(document_path)
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text().strip().replace("\n\n", "\n") + "\n"
        document = Document(page_content=content, metadata={"name": name})
        documents.append(document)

    print(f"Fetched {len(documents)} documents from knowledge base directory ({knowledge_base_dir_path})")
    return documents


def split_documents(documents: List[Document]) -> List[Chunk]:
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=75)
    splits = text_splitter.split_documents(documents=documents)
    for split in splits:
        source_document_name = split.metadata["name"]
        content = split.page_content
        chunk = Chunk(source_document_name=source_document_name, content=content)
        chunks.append(chunk)

    print(f"Divided {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def get_batches(
    ids: List[int],
    embeddings: List[List[float]],
    texts: List[str],
    metadatas: List[dict],
) -> Generator[Any]:
    start_idx = 0
    while len(ids[start_idx:]):
        end_idx = min(
            start_idx + CHROMA_DB_COLLECTION_MAX_BATCH_SIZE,
            len(ids),
        )
        batch_ids = ids[start_idx:end_idx]
        batch_embeddings = embeddings[start_idx:end_idx]
        batch_texts = texts[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]
        yield (batch_ids, batch_embeddings, batch_texts, batch_metadatas)
        start_idx = end_idx


def store_chunks(chunks: List[Chunk], chroma_client: ClientAPI, existing_collections: list) -> None:
    texts, metadatas, ids = [], [], []
    for chunk_id, chunk in enumerate(chunks):
        ids.append(str(chunk_id))
        texts.append(chunk.content)
        metadatas.append({"source_document_name": chunk.source_document_name})

    print(f"Processing total {len(chunks)} chunks")
    print("Proceeding to create vector embeddings")
    hf_embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings = hf_embedder.embed_documents(texts=texts)
    print(f"Created vector embeddings with each having {len(embeddings[0])} dimensions")

    if VECTOR_DB_COLLECTION_NAME in existing_collections:
        chroma_client.delete_collection(VECTOR_DB_COLLECTION_NAME)
        print("Deleted existing collection")

    collection = chroma_client.get_or_create_collection(VECTOR_DB_COLLECTION_NAME)
    batch_id = 1
    batches = get_batches(ids, embeddings, texts, metadatas)
    for ids, embeddings, documents, metadatas in batches:
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"Inserted batch {batch_id} ({len(ids)} elements)")
        batch_id += 1

    print(f"Vector store created with {collection.count()} documents")


def check_vector_store_existence() -> bool:
    db_exists = os.path.isdir(VECTOR_DB_PATH)
    if db_exists:
        chroma_client = PersistentClient(path=VECTOR_DB_PATH)
        collection_names = [c.name for c in chroma_client.list_collections()]
        return VECTOR_DB_COLLECTION_NAME in collection_names
    return False


def create_vector_store() -> None:
    print("Creating vector store")
    chroma_client = PersistentClient(path=VECTOR_DB_PATH)
    existing_collections = [c.name for c in chroma_client.list_collections()]
    print("Proceeding to scan documents and create vector embeddings")
    documents = fetch_documents(knowledge_base_dir_path=KNOWLEDGE_BASE_DIR_PATH)
    chunks = split_documents(documents=documents)
    store_chunks(
        chunks=chunks,
        chroma_client=chroma_client,
        existing_collections=existing_collections,
    )
    print("Created vector store and ingested documents")


def setup_vector_store() -> None:
    vector_store_exists = check_vector_store_existence()
    if not vector_store_exists:
        create_vector_store()
    else:
        print("Vector store already exists")
