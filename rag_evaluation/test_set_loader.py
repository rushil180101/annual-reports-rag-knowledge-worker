"""
This module provides helper functions to load the RAG test set.
"""

import json
from typing import List

from rag_evaluation.models import TestQuestion


def load_tests(test_set_file_path: str) -> List[TestQuestion]:
    test_questions = []
    with open(test_set_file_path) as test_set_file:
        for line in test_set_file.readlines():
            line = json.loads(line.strip())
            test_question = TestQuestion(**line)
            test_questions.append(test_question)
    return test_questions
