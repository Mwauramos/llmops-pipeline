# evaluator.py
# Custom LLM evaluation pipeline with MLflow tracking
# ZENIK.AI — Computational Epidemiology Portfolio

import os
import json
import datetime
import mlflow
from openai import OpenAI
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.test_questions import TEST_DATASET
from src.rag_pipeline import setup_rag_pipeline, query_with_metrics

load_dotenv()

FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def score_faithfulness(answer: str, ground_truth: str) -> float:
    """Score whether the answer correctly states the same fact as ground truth."""
    prompt = f"""Does the following answer correctly state the same information as the ground truth?

Ground truth: {ground_truth}

Answer: {answer}

Reply with a single number:
1 if the answer correctly states the same fact
0 if the answer is incorrect or contradicts the ground truth

Return only 0 or 1."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5


def score_relevancy(question: str, answer: str) -> float:
    """Score whether the answer is relevant to the question."""
    prompt = f"""Score whether this answer addresses the question asked.
Question: {question}
Answer: {answer}
Return only a number between 0 and 1. 1 means fully relevant, 0 means not relevant."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.5


def run_evaluation(query_engine) -> dict:
    """Run evaluation on the test dataset."""
    print(f"Running evaluation on {len(TEST_DATASET)} test questions...")

    faithfulness_scores = []
    relevancy_scores = []
    latencies = []
    costs = []

    for item in TEST_DATASET:
        result = query_with_metrics(query_engine, item["question"])

        f_score = score_faithfulness(result["answer"], item["ground_truth"])
        r_score = score_relevancy(item["question"], result["answer"])

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        latencies.append(result["latency_seconds"])
        costs.append(result["estimated_cost_usd"])

        print(f"  Q: {item['question'][:50]}...")
        print(f"     Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f}")

    metrics = {
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3),
        "answer_relevancy": round(sum(relevancy_scores) / len(relevancy_scores), 3),
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 2),
        "total_cost_usd": round(sum(costs), 6),
        "num_questions": len(TEST_DATASET),
        "timestamp": datetime.datetime.now().isoformat()
    }

    return metrics


def log_to_mlflow(metrics: dict):
    """Log evaluation metrics to MLflow."""
    mlflow.set_experiment("kenya-rag-evaluation")
    with mlflow.start_run(run_name="rag_evaluation"):
        mlflow.log_metric("faithfulness", metrics["faithfulness"])
        mlflow.log_metric("answer_relevancy", metrics["answer_relevancy"])
        mlflow.log_metric("avg_latency_seconds", metrics["avg_latency_seconds"])
        mlflow.log_metric("total_cost_usd", metrics["total_cost_usd"])
        mlflow.log_param("num_questions", metrics["num_questions"])
        mlflow.log_param("timestamp", metrics["timestamp"])
    print("Metrics logged to MLflow.")


def check_quality_gate(metrics: dict) -> bool:
    """Quality gate for CI/CD pipeline."""
    passes = (
        metrics["faithfulness"] >= FAITHFULNESS_THRESHOLD and
        metrics["answer_relevancy"] >= RELEVANCY_THRESHOLD
    )
    if passes:
        print("QUALITY GATE PASSED")
    else:
        print("QUALITY GATE FAILED — deployment blocked")
    print(f"  Faithfulness: {metrics['faithfulness']} (threshold: {FAITHFULNESS_THRESHOLD})")
    print(f"  Relevancy: {metrics['answer_relevancy']} (threshold: {RELEVANCY_THRESHOLD})")
    return passes


if __name__ == "__main__":
    print("Setting up RAG pipeline...")
    query_engine = setup_rag_pipeline()

    print("\nStarting evaluation...")
    metrics = run_evaluation(query_engine)

    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    log_to_mlflow(metrics)
    gate_passed = check_quality_gate(metrics)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_results.json", "w") as f:
        json.dump({**metrics, "gate_passed": gate_passed}, f, indent=2)

    print(f"\nResults saved to outputs/evaluation_results.json")
    if not gate_passed:
        exit(1)