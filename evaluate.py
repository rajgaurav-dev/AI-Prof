# evaluate.py

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


# ==============================
# LOAD TEST DATA
# ==============================
def load_test_data(file_path):

    data = []

    with open(file_path, "r") as f:

        for line in f:

            if line.strip() == "":
                continue

            question, ground_truth = line.strip().split("|")

            data.append({
                "question": question,
                "ground_truth": ground_truth
            })

    return data


# ==============================
# PRECISION@K
# ==============================
def precision_at_k(retriever, question, ground_truth, k=4):

    docs = retriever.invoke(question)

    retrieved_docs = docs[:k]

    relevant = 0

    for doc in retrieved_docs:
        if ground_truth.lower() in doc.page_content.lower():
            relevant += 1

    return relevant / k


# ==============================
# RECALL@K
# ==============================
def recall_at_k(retriever, question, ground_truth, k=4):

    docs = retriever.invoke(question)

    retrieved_docs = docs[:k]

    for doc in retrieved_docs:
        if ground_truth.lower() in doc.page_content.lower():
            return 1

    return 0


# ==============================
# LLM JUDGE
# ==============================
def llm_judge(chat_model, question, context, answer):

    prompt = f"""
You are an AI evaluator.

Question:
{question}

Context:
{context}

Answer:
{answer}

Score from 1-5:
1 = hallucinated
5 = correct and grounded

Return:

Score:
Explanation:
"""

    result = chat_model.invoke(prompt)

    return result.content


# ==============================
# MAIN EVALUATION FUNCTION
# ==============================
def run_rag_evaluation(retriever, rag_chain, chat_model, test_file):

    test_data = load_test_data(test_file)

    results = []
    precision_scores = []
    recall_scores = []

    for item in test_data:

        question = item["question"]
        gt = item["ground_truth"]

        docs = retriever.invoke(question)

        contexts = [doc.page_content for doc in docs]

        answer = rag_chain.invoke(question)

        precision = precision_at_k(retriever, question, gt)
        recall = recall_at_k(retriever, question, gt)

        precision_scores.append(precision)
        recall_scores.append(recall)

        judge = llm_judge(chat_model, question, "\n".join(contexts), answer)

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": gt
        })

        print("\n==============================")
        print("Question:", question)
        print("Answer:", answer)
        print("Judge:", judge)


    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    })


    ragas_result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )


    print("\n==============================")
    print("Retriever Metrics")
    print("==============================")

    print("Precision@k:", sum(precision_scores)/len(precision_scores))
    print("Recall@k:", sum(recall_scores)/len(recall_scores))


    print("\n==============================")
    print("RAGAS Metrics")
    print("==============================")

    print(ragas_result)

    return ragas_result