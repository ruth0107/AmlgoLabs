"""src/generator.py - Streaming response generation using mistral via Ollama."""

import sys
from pathlib import Path
from typing import List, Dict, Generator

import ollama
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from config import LLM_MODEL, NUM_CTX, TEMPERATURE, NUM_PREDICT

# Prompt Template
# Instruction-tuned format for Mistral (uses [INST] tokens internally via Ollama)
RAG_PROMPT = """\
You are a knowledgeable and precise assistant specializing in policy and legal documents.

Your task is to answer the user's question using ONLY the information from the provided context sections below.

Rules:
1. Base your answer strictly on the provided context.
2. If the context does not contain enough information, respond with: "The provided documents do not contain sufficient information to answer this question."
3. Do not fabricate or infer beyond what is explicitly stated.
4. Be clear, concise, and structured in your response.
5. If listing items, use bullet points for readability.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Answer:"""


def _format_context(chunks: List[Dict]) -> str:
    """Formats retrieved chunks into a numbered context block."""
    sections = []
    for i, chunk in enumerate(chunks, start=1):
        sections.append(f"[Section {i}]\n{chunk['text'].strip()}")
    return "\n\n".join(sections)


def stream_response(
    question: str,
    chunks: List[Dict],
) -> Generator[str, None, None]:
    """Streams a grounded response from Mistral via Ollama."""
    context = _format_context(chunks)
    prompt  = RAG_PROMPT.format(context=context, question=question)

    logger.info(f"Streaming response | model={LLM_MODEL} | ctx={NUM_CTX} | chunks={len(chunks)}")

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={
            "num_ctx":      NUM_CTX,
            "temperature":  TEMPERATURE,
            "num_predict":  NUM_PREDICT,
            "repeat_penalty": 1.1,
        },
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token


def get_no_context_response() -> Generator[str, None, None]:
    """Yield a canned response when no relevant chunks are found."""
    msg = (
        "I could not find relevant information in the provided documents "
        "to answer your question. Please try rephrasing or ask about a "
        "topic covered in the policy document."
    )
    for word in msg.split(" "):
        yield word + " "


if __name__ == "__main__":
    # Quick smoke test
    sample_chunks = [
        {
            "text": "We collect personal information such as your name, email address, "
                    "and usage data when you register or use our services.",
            "chunk_id": "chunk_0001",
            "score": 0.87,
        }
    ]
    query = "What personal information do you collect?"
    print(f"Query: {query}\nResponse: ", end="", flush=True)
    for token in stream_response(query, sample_chunks):
        print(token, end="", flush=True)
    print()
