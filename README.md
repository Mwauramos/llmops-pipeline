# Kenya RAG System — LLMOps Pipeline

Production monitoring and evaluation dashboard for the Kenya Health Surveillance RAG System. Tracks quality metrics, latency, cost, and deployment gate status using a custom evaluation pipeline with MLflow experiment tracking.

## Live Demo
https://llmops-pipeline-kogiqm8rpednrxgwjciu7t.streamlit.app/

## What it does
Evaluation Pipeline — runs 8 test questions through the RAG system and scores faithfulness and answer relevancy using GPT-4o-mini as an LLM judge

Quality Gate — blocks deployment if faithfulness drops below 70 percent or relevancy drops below 70 percent

Production Monitoring — tracks latency, cost, and SLA breaches per query with alerting

MLflow Tracking — every evaluation run is logged with full metrics for experiment comparison

## Tech Stack
LlamaIndex · Pinecone · OpenAI GPT-4o · MLflow · Streamlit · Python 3.11

## Results
Faithfulness: 100 percent
Answer Relevancy: 100 percent
Avg Latency: 2.51s
Cost per evaluation: $0.0016


GitHub: github.com/Mwauramos | Contact: mwauramos.n@gmail.com