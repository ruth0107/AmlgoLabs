"""
src/evaluation/dataset_generator.py
Generates a synthetic evaluation dataset (Questions and Ground Truth Answers)
by sampling random chunks from the document and using the LLM.
"""

import sys
import json
import random
from pathlib import Path

import ollama
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import LLM_MODEL, DATA_DIR
from src.chunker import load_chunks

EVAL_DATASET_PATH = DATA_DIR / "eval_dataset.json"

PROMPT_TEMPLATE = """
You are an expert reading comprehension assistant.
Given the following context from a policy document, generate a clear, specific question that can be answered ONLY using this context.
Then, provide the comprehensive ground truth answer.

Respond exactly in this JSON format (no markdown blocks, no other text):
{{"question": "<your question>", "ground_truth": "<your answer>"}}

Context:
{context}
"""

def generate_synthetic_dataset(num_samples: int = 5):
    """Samples chunks and generates Q&A pairs."""
    try:
        chunks = load_chunks()
    except FileNotFoundError:
        logger.error("Chunks not found. Run the app to build the index first.")
        return

    # Filter out very short chunks
    valid_chunks = [c for c in chunks if c.get("word_count", 0) > 40]
    
    if len(valid_chunks) < num_samples:
        logger.warning("Not enough valid chunks, using all available.")
        sampled = valid_chunks
    else:
        sampled = random.sample(valid_chunks, num_samples)

    dataset = []
    
    logger.info(f"Generating {len(sampled)} synthetic Q&A pairs using {LLM_MODEL}...")

    for i, chunk in enumerate(sampled, 1):
        logger.info(f"Generating pair {i}/{len(sampled)}...")
        
        prompt = PROMPT_TEMPLATE.format(context=chunk["text"])
        
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            
            content = response["message"]["content"].strip()
            # Clean up potential markdown JSON wrapping
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            qa_pair = json.loads(content)
            
            dataset.append({
                "question": qa_pair["question"],
                "ground_truth": qa_pair["ground_truth"],
                "source_chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"]
            })
        except Exception as e:
            logger.error(f"Failed to generate pair for chunk {chunk['chunk_id']}: {e}")

    with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    logger.success(f"Generated evaluation dataset saved to {EVAL_DATASET_PATH}")

if __name__ == "__main__":
    generate_synthetic_dataset(5)
