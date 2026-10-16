# app.py
# LLMOps Monitoring Dashboard
# ZENIK.AI — Computational Epidemiology Portfolio

import streamlit as st
import json
import os
import sys
import pandas as pd
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

st.set_page_config(
    page_title="Kenya RAG LLMOps Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Kenya RAG System — LLMOps Dashboard")
st.markdown(
    "Production monitoring and evaluation dashboard for the Kenya Health "
    "Surveillance RAG System. Tracks quality metrics, latency, cost, and "
    "quality gate status."
)
st.divider()

tab1, tab2, tab3 = st.tabs(["Evaluation Results", "Monitoring", "Run Evaluation"])

with tab1:
    st.markdown("### Latest Evaluation Results")

    eval_path = "outputs/evaluation_results.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score = eval_results["faithfulness"]
            st.metric("Faithfulness", f"{score:.1%}", delta="Pass" if score >= 0.7 else "Fail")
        with col2:
            score = eval_results["answer_relevancy"]
            st.metric("Answer Relevancy", f"{score:.1%}", delta="Pass" if score >= 0.7 else "Fail")
        with col3:
            st.metric("Avg Latency", f"{eval_results['avg_latency_seconds']}s")
        with col4:
            st.metric("Total Cost", f"${eval_results['total_cost_usd']:.4f}")

        gate = eval_results.get("gate_passed", False)
        if gate:
            st.success("QUALITY GATE PASSED — System approved for deployment")
        else:
            st.error("QUALITY GATE FAILED — Deployment blocked")

        st.markdown(f"**Last evaluated:** {eval_results['timestamp']}")
        st.markdown(f"**Questions evaluated:** {eval_results['num_questions']}")
    else:
        st.info("No evaluation results yet. Run the evaluation from the Run Evaluation tab.")

with tab2:
    st.markdown("### Production Monitoring")

    monitor_path = "outputs/monitoring_results.json"
    if os.path.exists(monitor_path):
        with open(monitor_path) as f:
            monitoring = json.load(f)

        df = pd.DataFrame(monitoring)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Latency", f"{df['latency_seconds'].mean():.2f}s")
        with col2:
            st.metric("Total Cost", f"${df['estimated_cost_usd'].sum():.4f}")
        with col3:
            total_alerts = sum(len(m["alerts"]) for m in monitoring)
            st.metric("Active Alerts", total_alerts)

        st.markdown("### Query Log")
        for item in monitoring:
            with st.expander(f"Query: {item['question'][:60]}"):
                st.markdown(f"**Timestamp:** {item['timestamp']}")
                st.markdown(f"**Latency:** {item['latency_seconds']}s")
                st.markdown(f"**Cost:** ${item['estimated_cost_usd']}")
                st.markdown(f"**Sources retrieved:** {item['num_sources']}")
                if item["alerts"]:
                    for alert in item["alerts"]:
                        st.warning(alert)
                else:
                    st.success("No alerts")
    else:
        st.info("No monitoring data yet.")

with tab3:
    st.markdown("### Run Evaluation")
    st.markdown(
        "Run the evaluation pipeline against the Kenya health knowledge base. "
        "Results are logged to MLflow and saved locally."
    )

    if st.button("Run Evaluation Now", type="primary"):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()

        with st.spinner("Setting up RAG pipeline..."):
            from rag_pipeline import setup_rag_pipeline, query_with_metrics
            from data.test_questions import TEST_DATASET
            query_engine = setup_rag_pipeline()

        st.success("RAG pipeline ready.")
        progress = st.progress(0)

        client_eval = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        f_scores = []
        r_scores = []
        latencies = []
        costs = []

        with st.spinner("Running evaluation..."):
            for i, item in enumerate(TEST_DATASET):
                result = query_with_metrics(query_engine, item["question"])

                prompt_f = f"Does this answer correctly state the same information as the ground truth?\nGround truth: {item['ground_truth']}\nAnswer: {result['answer']}\nReturn only 0 or 1."
                resp_f = client_eval.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_f}],
                    temperature=0
                )
                f_score = float(resp_f.choices[0].message.content.strip())

                prompt_r = f"Score whether this answer addresses the question.\nQuestion: {item['question']}\nAnswer: {result['answer']}\nReturn only a number 0 to 1."
                resp_r = client_eval.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_r}],
                    temperature=0
                )
                r_score = float(resp_r.choices[0].message.content.strip())

                f_scores.append(f_score)
                r_scores.append(r_score)
                latencies.append(result["latency_seconds"])
                costs.append(result["estimated_cost_usd"])
                progress.progress((i + 1) / len(TEST_DATASET))

        metrics = {
            "faithfulness": round(sum(f_scores) / len(f_scores), 3),
            "answer_relevancy": round(sum(r_scores) / len(r_scores), 3),
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 2),
            "total_cost_usd": round(sum(costs), 6),
            "num_questions": len(TEST_DATASET),
            "timestamp": datetime.datetime.now().isoformat(),
            "gate_passed": True
        }

        gate_passed = metrics["faithfulness"] >= 0.7 and metrics["answer_relevancy"] >= 0.7
        metrics["gate_passed"] = gate_passed

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/evaluation_results.json", "w") as f_out:
            json.dump(metrics, f_out, indent=2)

        if gate_passed:
            st.success("QUALITY GATE PASSED")
        else:
            st.error("QUALITY GATE FAILED")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Faithfulness", f"{metrics['faithfulness']:.1%}")
        with col2:
            st.metric("Answer Relevancy", f"{metrics['answer_relevancy']:.1%}")

        st.rerun()

st.divider()
st.caption("ZENIK.AI · Kenya RAG LLMOps Dashboard · MLflow + Custom Evaluation Pipeline")