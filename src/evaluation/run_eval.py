"""
src/evaluation/run_eval.py
Orchestrates the evaluation pipeline.
Loads the test dataset, runs queries through the RAG pipeline, scores them,
and generates a final markdown report.
"""

import sys
import json
import statistics
from pathlib import Path

# --- HACK: Prevent chromadb ONNX DLL crash on Windows ---
try:
    import chromadb.utils.embedding_functions as ef
    ef.DefaultEmbeddingFunction = lambda: None
except Exception:
    pass
# --------------------------------------------------------

from loguru import logger

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_DIR
from src.pipeline import stream_answer
from src.evaluation.dataset_generator import EVAL_DATASET_PATH
from src.evaluation.metrics import (
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_groundedness,
    evaluate_hallucination_rate
)

REPORT_PATH = DATA_DIR / "rag_evaluation_report.md"

def run_evaluation():
    if not EVAL_DATASET_PATH.exists():
        logger.error(f"Dataset not found at {EVAL_DATASET_PATH}. Run dataset_generator.py first.")
        return
        
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    logger.info(f"Starting evaluation on {len(dataset)} queries...")
    
    results = []
    
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        logger.info(f"\n[{i}/{len(dataset)}] Evaluating Query: {question}")
        
        # 1. Run RAG Pipeline
        token_stream, sources = stream_answer(question)
        
        # Collect the full generated answer
        generated_answer = "".join([token for token in token_stream])
        
        # 2. Score Metrics
        precision = evaluate_context_precision(question, sources)
        recall = evaluate_context_recall(ground_truth, sources)
        groundedness = evaluate_groundedness(generated_answer, sources)
        hallucination = evaluate_hallucination_rate(groundedness)
        
        logger.info(f"  Precision: {precision:.2f} | Recall: {recall:.2f} | Groundedness: {groundedness:.2f}")
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "groundedness": groundedness,
                "hallucination": hallucination
            }
        })
        
    # 3. Aggregate and Generate Report
    avg_precision = statistics.mean([r["metrics"]["precision"] for r in results])
    avg_recall = statistics.mean([r["metrics"]["recall"] for r in results])
    avg_groundedness = statistics.mean([r["metrics"]["groundedness"] for r in results])
    avg_hallucination = statistics.mean([r["metrics"]["hallucination"] for r in results])
    
    report_md = f"""# RAG Pipeline Evaluation Report

## Aggregate Metrics

| Metric | Score | Explanation |
|---|---|---|
| **Context Precision** | `{avg_precision:.2f}` | Proportion of retrieved chunks that are relevant. |
| **Context Recall** | `{avg_recall:.2f}` | How well the retrieved context covers the ground truth. |
| **Groundedness** | `{avg_groundedness:.2f}` | How faithful the generated answer is to the retrieved context. |
| **Hallucination Rate** | `{avg_hallucination:.2f}` | Rate of fabricated facts not in context. |

---

## Detailed Results

"""
    for i, res in enumerate(results, 1):
        report_md += f"### Query {i}: {res['question']}\n\n"
        report_md += f"- **Ground Truth:** {res['ground_truth']}\n"
        report_md += f"- **Generated Answer:** {res['generated_answer']}\n\n"
        report_md += "**Scores:**\n"
        report_md += f"- Precision: `{res['metrics']['precision']:.2f}`\n"
        report_md += f"- Recall: `{res['metrics']['recall']:.2f}`\n"
        report_md += f"- Groundedness: `{res['metrics']['groundedness']:.2f}`\n\n"
        report_md += "---\n\n"
        
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    logger.success(f"Evaluation complete. Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run_evaluation()
