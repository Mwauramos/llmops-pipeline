# monitor.py
# Production monitoring dashboard data collector
# ZENIK.AI — Computational Epidemiology Portfolio

import os
import json
import datetime
import mlflow
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.rag_pipeline import setup_rag_pipeline, query_with_metrics

load_dotenv()

LATENCY_SLA = 5.0
COST_BUDGET_PER_QUERY = 0.01


def monitor_single_query(query_engine, question: str) -> dict:
    """Run a query and collect monitoring metrics."""
    result = query_with_metrics(query_engine, question)

    alerts = []
    if result["latency_seconds"] > LATENCY_SLA:
        alerts.append(f"LATENCY BREACH: {result['latency_seconds']}s > {LATENCY_SLA}s SLA")
    if result["estimated_cost_usd"] > COST_BUDGET_PER_QUERY:
        alerts.append(f"COST BREACH: ${result['estimated_cost_usd']} > ${COST_BUDGET_PER_QUERY} budget")

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "latency_seconds": result["latency_seconds"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "answer_length": len(result["answer"]),
        "num_sources": len(result["source_nodes"]),
        "alerts": alerts
    }


def log_monitoring_run(metrics: list):
    """Log monitoring metrics to MLflow."""
    mlflow.set_experiment("kenya-rag-monitoring")

    avg_latency = sum(m["latency_seconds"] for m in metrics) / len(metrics)
    total_cost = sum(m["estimated_cost_usd"] for m in metrics)
    total_alerts = sum(len(m["alerts"]) for m in metrics)

    with mlflow.start_run(run_name="monitoring_run"):
        mlflow.log_metric("avg_latency_seconds", round(avg_latency, 2))
        mlflow.log_metric("total_cost_usd", round(total_cost, 6))
        mlflow.log_metric("total_alerts", total_alerts)
        mlflow.log_metric("queries_monitored", len(metrics))

    print(f"Monitoring run logged to MLflow.")
    print(f"  Avg latency: {avg_latency:.2f}s")
    print(f"  Total cost: ${total_cost:.6f}")
    print(f"  Alerts: {total_alerts}")


if __name__ == "__main__":
    print("Starting monitoring run...")
    query_engine = setup_rag_pipeline()

    test_queries = [
        "What is the malaria prevalence in Kenya?",
        "What is the HIV situation in Kenya?",
        "What are the TB treatment outcomes in Kenya?"
    ]

    results = []
    for q in test_queries:
        result = monitor_single_query(query_engine, q)
        results.append(result)
        print(f"Query: {q[:50]}")
        print(f"  Latency: {result['latency_seconds']}s")
        print(f"  Cost: ${result['estimated_cost_usd']}")
        if result["alerts"]:
            for alert in result["alerts"]:
                print(f"  ALERT: {alert}")

    log_monitoring_run(results)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/monitoring_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Monitoring complete.")