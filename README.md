---
title: RAG knowledge worker / chat assistant (Financial Annual Reports data)
emoji: 📚
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
---

# RAG knowledge worker / chat assistant for annual reports

This project implements RAG based chat assistant for answering questions related to annual reports data.

## How to execute
- Install dependencies

    `pip install -r requirements.txt`

- Execute `app.py`

    `python app.py`

- This will open a gradio chat interface, and also setup the vectore store if it does not exist.

-----

## How to evaluate RAG performance (retrieval evaluation + answer evaluation)?

Note: Before evaluating, the vector store should already be setup.
- Open `rag_evaluation/test_set.jsonl`

- Add test samples

- Execute `evaluate.py`

    `python evaluate.py`

- This will run retrieval evaluation and answer evaluation for the given test sample and print the average of following metrics.

    - Retrieval metrics (in the range 0-100)
        - Mean reciprocal rank (MRR)
        - DCG (Discounted Cumulative Gain)
        - nDCG (Normalized DCG)

    - Answer metrics (in the range 0-5)
        - Relevance
        - Accuracy
        - Completeness

-----

## References

Annual reports data downloaded from here: https://www.annualreports.com/Company/motorsport-games-inc

-----