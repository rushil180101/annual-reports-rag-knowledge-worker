"""
This module is the interface for interacting with RAG application.
"""

import gradio as gr

from answer import chat
from ingest import setup_vector_store

if __name__ == "__main__":
    setup_vector_store()
    demo = gr.ChatInterface(
        fn=chat,
        title="📚 RAG Chat Assistant for Annual Reports",
    )
    demo.launch(inbrowser=True)
