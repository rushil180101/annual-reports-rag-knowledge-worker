"""
This module defines the models for RAG evaluation.
"""

from typing import List

from pydantic import BaseModel, Field


class TestQuestion(BaseModel):
    question: str = Field(description="Test question to evaluate RAG application")
    reference_answer: str = Field(description="Reference answer to the test question to evaluate RAG application")
    keywords: List[str] = Field(description="Expected keywords in the answer to evaluate RAG application")


class RetrievalEvaluation(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank to evaluate RAG document retrievals")
    dcg: float = Field(description="Discounted Cumulative Gain to evaluate RAG document retrievals")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain to evaluate RAG document retrievals")


class AnswerEvaluation(BaseModel):
    feedback: str = Field(
        description="Feedback on how the answer produced by RAG application compares with the actual answer"
    )
    relevance: float = Field(
        description=(
            "Relevance score from 0 to 10, indicating how relevant the answer produced by RAG application is, "
            "compared to the reference answer"
        )
    )
    accuracy: float = Field(
        description=(
            "Accuracy score from 0 to 10, indicating how accurate the answer produced by RAG application is, "
            "compared to the reference answer"
        )
    )
    completeness: float = Field(
        description=(
            "Completeness score from 0 to 10, indicating how complete the answer produced by RAG application is, "
            "compared to the reference answer"
        )
    )
