"""
This module defines the helper functions for LLM interaction and data processing.
"""

import json
import os
from typing import List
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
from litellm import completion
from common.models import Chunk
from common.constants import (
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    VECTOR_DB_COLLECTION_NAME,
    EMBEDDINGS_RETRIEVAL_COUNT,
    CHUNKS_RERANKING_MODEL,
    CHAT_MODEL,
)

CHAT_ASSISTANT_SYSTEM_PROMPT = """
You are a helpful chat assistant responsible for answering
user's questions. If you don't know the answer, say so.
"""


def check_vector_store_existence() -> bool:
    db_exists = os.path.isdir(VECTOR_DB_PATH)
    if db_exists:
        chroma_client = PersistentClient(path=VECTOR_DB_PATH)
        collection_names = [c.name for c in chroma_client.list_collections()]
        return VECTOR_DB_COLLECTION_NAME in collection_names
    return False


def get_relevant_chunks(question: str) -> List[Chunk]:
    print("Proceeding to fetch relevant chunks from vector store")
    chroma_client = PersistentClient(path=VECTOR_DB_PATH)
    collection = chroma_client.get_collection(VECTOR_DB_COLLECTION_NAME)
    hf_embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    query_embedding = hf_embedder.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=EMBEDDINGS_RETRIEVAL_COUNT,
        include=["documents", "metadatas"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    chunks = []
    for content, metadata in zip(documents, metadatas):
        chunk = Chunk(
            source_document_name=metadata["source_document_name"],
            content=content,
        )
        chunks.append(chunk)

    print(f"Fetched {len(chunks)} relevant chunks from vector store")
    return chunks


def rerank_chunks(question: str, chunks: List[Chunk]) -> List[Chunk]:
    print(f"Proceeding to rerank {len(chunks)} chunks")
    chunks_reranking_system_prompt = """
        You are a document re-ranker responsible for re-ranking the document
        chunks based on the relevance of a given question. You will be given a
        question and a list of chunks of text, along with the ids of each chunk.
        Your job is to re-rank or sort the chunk ids based on relevance to the
        question, with the most relevant chunk being ranked first and the least
        relevant chunk being ranked last. You should return the list of all chunk
        ids, re-ranked in the order from most relevant to the least relevant.
        Return only the list of chunk ids.
        """
    chunks_reranking_user_prompt = """
        Re-order the following document chunks in the order of their relevance
        to the question. Respond only with the list of all reordered chunks ids.        
        """
    chunks_reranking_user_prompt += f"Question: {question}\n"
    for chunk_id, chunk in enumerate(chunks):
        chunks_reranking_user_prompt += f"Chunk id: {chunk_id}\n" + f"Chunk content: {chunk.content}\n\n"
    messages = [
        {"role": "system", "content": chunks_reranking_system_prompt},
        {"role": "user", "content": chunks_reranking_user_prompt},
    ]

    response = completion(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=CHUNKS_RERANKING_MODEL,
        messages=messages,
    )
    reranked_chunk_ids = json.loads(response.choices[0].message.content)
    assert len(reranked_chunk_ids) == len(chunks)
    print(f"Reranked {len(reranked_chunk_ids)} chunks: {reranked_chunk_ids}")
    return reranked_chunk_ids


def chat(question: str, history: List[dict]) -> str:
    chunks = get_relevant_chunks(question=question)
    reranked_chunk_ids = rerank_chunks(question=question, chunks=chunks)
    reranked_chunks = []
    for chunk_id in reranked_chunk_ids:
        reranked_chunks.append(chunks[chunk_id])

    context = "\n".join([chunk.content for chunk in reranked_chunks])
    user_prompt = f"""
        Please answer the following question based on the provided context
        Question: {question}
        Context: {context}
        """
    messages = (
        [{"role": "system", "content": CHAT_ASSISTANT_SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_prompt}]
    )

    response = completion(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=CHAT_MODEL,
        messages=messages,
    )
    llm_reply = response.choices[0].message.content
    return llm_reply
