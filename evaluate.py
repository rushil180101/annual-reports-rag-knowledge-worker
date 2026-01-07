"""
This module evaluates the performance of RAG application.
"""

from pydantic import BaseModel

from answer import chat
from rag_evaluation.answer_evals import evaluate_answer
from rag_evaluation.retrieval_evals import evaluate_retrieval
from rag_evaluation.test_set_loader import load_tests

TEST_SET_FILE_PATH = "rag_evaluation/test_set.jsonl"


class TestSetEvaluationResult(BaseModel):
    avg_mrr: float
    avg_dcg: float
    avg_ndcg: float
    avg_relevance: float
    avg_accuracy: float
    avg_completeness: float


def get_test_set_evaluation_result() -> TestSetEvaluationResult:
    test_questions = load_tests(test_set_file_path=TEST_SET_FILE_PATH)

    # retrieval metrics
    mrr = 0
    dcg = 0
    ndcg = 0

    # answer metrics
    relevance = 0
    accuracy = 0
    completeness = 0

    total_count = len(test_questions)

    for count, test_question in enumerate(test_questions, start=1):
        print(f"Evaluating test sample {count:>3} / {total_count:>3}")

        print("Evaluating retrieval")
        retrieval_evaluation = evaluate_retrieval(test_question=test_question)
        mrr += retrieval_evaluation.mrr
        dcg += retrieval_evaluation.dcg
        ndcg += retrieval_evaluation.ndcg

        print("Evaluating answer")
        rag_answer = chat(question=test_question.question, history=[])
        answer_evaluation = evaluate_answer(
            test_question=test_question,
            rag_answer=rag_answer,
        )
        relevance += answer_evaluation.relevance
        accuracy += answer_evaluation.accuracy
        completeness += answer_evaluation.completeness

        print(f"Evaluated test sample {count:>3} / {total_count:>3}")
        print()

    avg_mrr = mrr / total_count
    avg_dcg = dcg / total_count
    avg_ndcg = ndcg / total_count
    avg_relevance = relevance / total_count
    avg_accuracy = accuracy / total_count
    avg_completeness = completeness / total_count

    print(f"Evaluated total {total_count} samples")

    result = TestSetEvaluationResult(
        avg_mrr=avg_mrr,
        avg_dcg=avg_dcg,
        avg_ndcg=avg_ndcg,
        avg_relevance=avg_relevance,
        avg_accuracy=avg_accuracy,
        avg_completeness=avg_completeness,
    )
    return result


if __name__ == "__main__":
    result = get_test_set_evaluation_result()
    print("RAG test set evaluation results")
    print(f"avg_mrr = {result.avg_mrr * 100} / 100")
    print(f"avg_dcg = {result.avg_dcg * 100} / 100")
    print(f"avg_ndcg = {result.avg_ndcg * 100} / 100")
    print(f"avg_relevance = {result.avg_relevance} / 5")
    print(f"avg_accuracy = {result.avg_accuracy} / 5")
    print(f"avg_completeness = {result.avg_completeness} / 5")
