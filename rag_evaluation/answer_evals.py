"""
This module defines the answer evaluation functions for RAG application.
"""

from litellm import completion

from common.constants import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, RAG_ANSWER_EVALUATION_MODEL
from rag_evaluation.models import AnswerEvaluation, TestQuestion

SYSTEM_PROMPT = """
    You are an intelligent judge responsible for judging answers produced by a chatbot.
    You will be given a question, the answer of this question produced by a chatbot,
    and a reference answer as a ground truth. Evaluate the answer produced by chatbot
    by comparing it with the reference answer, and assign the following scores ranging
    from 0 to 5, with 5 being the highest score and 0 being the lowest score.
    Relevance score: Indicates how relevant the chatbot's answer is.
    Accuracy: Indicates how accurate the chatbot's answer is.
    Completeness: Indicates how complete the chatbot's answer is.
    Feedback: Your short feedaback about the chatbot's response, suggest improvements.
    Respond only with feedback, relevance, accuracy, completeness.
    """


def evaluate_answer(test_question: TestQuestion, rag_answer) -> AnswerEvaluation:
    question = test_question.question
    reference_answer = test_question.reference_answer
    user_prompt = f"""
        Score the following response of chatbot based on relevance, accuracy, completeness.
        Question: {question}
        Chatbot's answer: {rag_answer}
        Reference answer (ground truth): {reference_answer}
        Respond only with feedback, relevance, accuracy, completeness in the following json format.
        {{"feedback": "...", "relevance": 4.3, "accuracy": 4.2, "completeness": 3.7}}
        """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    response = completion(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=RAG_ANSWER_EVALUATION_MODEL,
        messages=messages,
        response_format=AnswerEvaluation,
    )
    json_data = response.choices[0].message.content
    answer_evaluation = AnswerEvaluation.model_validate_json(json_data)
    return answer_evaluation
