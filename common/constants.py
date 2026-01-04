"""
This module defines the constants for RAG application.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Openrouter credentials
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
