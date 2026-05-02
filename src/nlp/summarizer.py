"""
Medical Report Summarizer Module
---------------------------------
Uses Facebook BART-Large-CNN for abstractive summarization.
Handles edge cases: empty input, very short text, and token limits.
Includes extractive fallback when model is unavailable.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging

logger = logging.getLogger(__name__)

# ── Model initialization ──
MODEL_NAME = "facebook/bart-large-cnn"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    MODEL_LOADED = True
    logger.info("✅ BART summarization model loaded successfully.")
except Exception as e:
    summarizer = None
    MODEL_LOADED = False
    logger.warning(f"⚠️ Summarization model failed to load: {e}")


def summarize_report(text):
    """
    Generate an abstractive summary of a radiology report.

    Args:
        text (str): Full radiology report text.

    Returns:
        str: Summarized text or fallback message.
    """
    if not text or not text.strip():
        return "No input provided."

    # Check for OCR error messages
    if text.startswith("[OCR Error]") or text.startswith("[OCR Warning]"):
        return text

    if not MODEL_LOADED or summarizer is None:
        return _extractive_fallback(text)

    # Handle very short texts (BART needs minimum input)
    word_count = len(text.split())
    if word_count < 10:
        return text.strip()

    try:
        # Calculate dynamic lengths based on input
        max_len = max(30, min(150, int(word_count * 0.6)))
        min_len = max(10, min(50, int(word_count * 0.3)))

        # Truncate to BART's max input (1024 tokens)
        tokens = tokenizer.encode(text, truncation=True, max_length=1024)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

        result = summarizer(
            truncated_text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
        )
        return result[0]["summary_text"]

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return _extractive_fallback(text)


def _extractive_fallback(text):
    """Simple extractive fallback if model is unavailable."""
    sentences = text.replace("\n", " ").split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    if not sentences:
        return text[:500]
    # Return first 3 most informative sentences
    return ". ".join(sentences[:3]) + "."