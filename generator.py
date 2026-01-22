# Use a pipeline as a high-level helper
from transformers import pipeline
import regex as re
pipe = pipeline("text-generation", model="Qwen/Qwen3-1.7B")

import logging

def generate_answer(query: str, contexts: list) -> str:
    try:
        message = f"""
Answer the question using the context below.

Context:
{chr(10).join(contexts)}

Question:
{query}

Answer:
"""

        response = pipe(message)

        # 1️⃣ Safely extract text
        generated_text = response[0].get("generated_text", "")
        if not generated_text:
            logging.warning("Empty generated text")
            return "No answer generated."

        # 2️⃣ Regex-based extraction (SAFE)
        match = re.search(
            r"Answer:\s*(.*?)(?:\nOkay,|\nNote:|\Z)",
            generated_text,
            re.S | re.I
        )

        if match:
            return match.group(1).strip()

        # 3️⃣ Fallback if format is unexpected
        logging.warning("Answer marker not found, returning full text")
        return generated_text.strip()

    except Exception as e:
        logging.error(f"Answer generation failed: {e}")
        return "Sorry, something went wrong while generating the answer."