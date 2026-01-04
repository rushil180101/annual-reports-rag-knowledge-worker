"""
This module defines the models for RAG application.
"""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    source_document_name: str = Field(description="Source document name from which this chunk is extracted")
    content: str = Field(description="Content of the chunk")
