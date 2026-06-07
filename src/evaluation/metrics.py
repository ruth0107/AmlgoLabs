"""
src/evaluation/metrics.py
LLM-as-a-judge scorers for RAG evaluation metrics:
- Context Precision
- Context Recall
- Groundedness
- Hallucination Rate (1 - Groundedness)
"""

import sys
from pathlib import Path
from typing import List, Dict

import ollama
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import LLM_MODEL

def _ask_judge(prompt: str) -> str:
    """Helper to query the LLM judge."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}  # Deterministic scoring
        )
        return response["message"]["content"].strip().upper()
    except Exception as e:
        logger.error(f"LLM Judge failed: {e}")
        return "ERROR"

def evaluate_context_precision(question: str, retrieved_chunks: List[Dict]) -> float:
    """
    Context Precision: Are the retrieved chunks relevant to the question?
    Returns: Proportion of retrieved chunks that are relevant [0.0 - 1.0].
    """
    if not retrieved_chunks:
        return 0.0
        
    relevant_count = 0
    
    for chunk in retrieved_chunks:
        prompt = f"""
        Given the question: "{question}"
        Is the following context chunk relevant to answering the question?
        Context: "{chunk['text']}"
        
        Answer ONLY with "YES" or "NO".
        """
        result = _ask_judge(prompt)
        if "YES" in result:
            relevant_count += 1
            
    return relevant_count / len(retrieved_chunks)

def evaluate_context_recall(ground_truth: str, retrieved_chunks: List[Dict]) -> float:
    """
    Context Recall: Did we retrieve all information needed for the ground truth?
    Since we do statement extraction implicitly, we ask the judge if the ground truth
    can be completely deduced from the combined context.
    Returns: 1.0 if fully deducible, 0.5 if partially, 0.0 if not at all.
    """
    if not retrieved_chunks:
        return 0.0
        
    combined_context = "\n\n".join([c['text'] for c in retrieved_chunks])
    
    prompt = f"""
    Given the following retrieved context, can the ground truth statement be completely deduced from it?
    
    Context:
    {combined_context}
    
    Ground Truth: "{ground_truth}"
    
    Answer ONLY with "FULL" if it is completely supported, "PARTIAL" if only partially supported, or "NONE" if not supported at all.
    """
    
    result = _ask_judge(prompt)
    
    if "FULL" in result:
        return 1.0
    elif "PARTIAL" in result:
        return 0.5
    return 0.0

def evaluate_groundedness(generated_answer: str, retrieved_chunks: List[Dict]) -> float:
    """
    Groundedness: Is the generated answer strictly based on the retrieved context?
    Instead of complex sentence splitting, we ask the judge to rate the overall faithfulness.
    Returns: 1.0 (Fully Grounded), 0.5 (Partially Hallucinated), 0.0 (Completely Hallucinated).
    """
    if not retrieved_chunks:
        return 0.0
        
    combined_context = "\n\n".join([c['text'] for c in retrieved_chunks])
    
    prompt = f"""
    You are evaluating whether an AI's generated answer is grounded in the provided context.
    
    Context:
    {combined_context}
    
    Generated Answer: "{generated_answer}"
    
    Does the generated answer contain any information, claims, or facts that are NOT explicitly stated in the context?
    
    Answer ONLY with "FULLY_GROUNDED" (no outside facts), "PARTIALLY_GROUNDED" (mix of context and outside facts), or "NOT_GROUNDED" (mostly outside facts/hallucinations).
    """
    
    result = _ask_judge(prompt)
    
    if "FULLY_GROUNDED" in result:
        return 1.0
    elif "PARTIALLY_GROUNDED" in result:
        return 0.5
    return 0.0

def evaluate_hallucination_rate(groundedness_score: float) -> float:
    """Hallucination rate is the exact inverse of groundedness."""
    return 1.0 - groundedness_score
