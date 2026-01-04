"""
This module is the interface for interacting with RAG application.
"""

import gradio as gr
from answer import (
    check_vector_store_existence,
    chat,
)

if __name__ == "__main__":
    vector_store_exists = check_vector_store_existence()
    if not vector_store_exists:
        raise Exception("Vector store does not exist. Please setup the vector store first.")

    demo = gr.ChatInterface(
        fn=chat,
        title="📚 RAG Chat Assistant",
        description="Ask questions over your document knowledge base",
    )
    demo.launch(inbrowser=True)
