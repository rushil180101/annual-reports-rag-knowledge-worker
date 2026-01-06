"""
This module defines the retrieval evaluation functions for RAG application.
"""

from math import log2
from typing import List

from answer import get_relevant_chunks
from common.models import Chunk
from rag_evaluation.models import RetrievalEvaluation, TestQuestion

MRR_RELEVANCE_SCORE_THRESHOLD = 2


def map_relevance_score_to_int(relevance_score: float) -> int:
    # Map relevance score to integer representation
    result = 0
    if relevance_score >= 0.75:
        result = 3
    elif relevance_score >= 0.50:
        result = 2
    elif relevance_score >= 0.25:
        result = 1
    return result


def get_relevance_scores(test_question: TestQuestion, retrieved_chunks: List[Chunk]) -> List[float]:
    relevance_scores = []
    keywords = [keyword.lower() for keyword in test_question.keywords]
    total_keywords = len(keywords)
    for chunk in retrieved_chunks:
        match_keywords = 0
        chunk_content = chunk.content.lower()
        for keyword in keywords:
            if keyword in chunk_content:
                match_keywords += 1
        relevance_score = match_keywords / total_keywords
        relevance_score = map_relevance_score_to_int(relevance_score)
        relevance_scores.append(relevance_score)
    return relevance_scores


def calculate_mrr(test_question: TestQuestion, retrieved_chunks: List[Chunk]) -> float:
    relevance_scores = get_relevance_scores(
        test_question=test_question,
        retrieved_chunks=retrieved_chunks,
    )
    for rank, relevance_score in enumerate(relevance_scores, start=1):
        if relevance_score >= MRR_RELEVANCE_SCORE_THRESHOLD:
            return 1 / rank
    return 0


def calculate_dcg(test_question: TestQuestion, retrieved_chunks: List[Chunk]) -> float:
    dcg = 0
    relevance_scores = get_relevance_scores(
        test_question=test_question,
        retrieved_chunks=retrieved_chunks,
    )
    for rank, score in enumerate(relevance_scores, start=1):
        dcg += score / log2(rank + 1)
    return dcg


def calculate_ndcg(test_question: TestQuestion, retrieved_chunks: List[Chunk]) -> float:
    dcg = calculate_dcg(test_question=test_question, retrieved_chunks=retrieved_chunks)
    relevance_scores = get_relevance_scores(
        test_question=test_question,
        retrieved_chunks=retrieved_chunks,
    )
    sorted_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = 0
    for rank, score in enumerate(sorted_relevance_scores, start=1):
        idcg += score / log2(rank + 1)

    ndcg = (dcg / idcg) if idcg > 0 else 0
    return ndcg


def evaluate_retrieval(test_question: TestQuestion) -> RetrievalEvaluation:
    question = test_question.question
    chunks = get_relevant_chunks(question=question)
    mrr = calculate_mrr(test_question=test_question, retrieved_chunks=chunks)
    dcg = calculate_dcg(test_question=test_question, retrieved_chunks=chunks)
    ndcg = calculate_ndcg(test_question=test_question, retrieved_chunks=chunks)
    retrieval_evaluation = RetrievalEvaluation(mrr=mrr, dcg=dcg, ndcg=ndcg)
    return retrieval_evaluation
