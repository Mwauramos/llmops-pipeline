# rag_pipeline.py
# RAG pipeline for the LLMOps evaluation system
# Reuses the Kenya health knowledge base from Project 1
# ZENIK.AI — Computational Epidemiology Portfolio

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()


def setup_rag_pipeline():
    """Connect to existing Pinecone index and set up query engine."""
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    Settings.llm = OpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("kenya-surveillance")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )

    query_engine = index.as_query_engine(
        similarity_top_k=6,
        response_mode="tree_summarize"
    )

    return query_engine


def query_with_metrics(query_engine, question: str) -> dict:
    """Run a query and capture latency and cost metrics."""
    start_time = time.time()

    response = query_engine.query(question)

    latency = round(time.time() - start_time, 2)

    # Estimate token cost (approximate)
    response_text = str(response)
    estimated_tokens = len(response_text.split()) * 1.3
    estimated_cost = round(estimated_tokens * 0.000015, 6)

    return {
        "question": question,
        "answer": response_text,
        "latency_seconds": latency,
        "estimated_cost_usd": estimated_cost,
        "source_nodes": [
            {
                "text": node.text[:200],
                "score": round(node.score, 3) if node.score else 0
            }
            for node in response.source_nodes
        ]
    }


if __name__ == "__main__":
    print("Setting up RAG pipeline...")
    qe = setup_rag_pipeline()
    result = query_with_metrics(qe, "What is the malaria prevalence in Kenya?")
    print(f"Answer: {result['answer'][:200]}")
    print(f"Latency: {result['latency_seconds']}s")
    print(f"Estimated cost: ${result['estimated_cost_usd']}")
    print("RAG pipeline working.")