"""
Text Summarizer using Hugging Face Transformers.

This module provides functionality to summarize text using a pretrained
transformer model from Hugging Face, optimized for CPU-only execution.
"""

import os
from typing import Optional

try:
    from transformers import pipeline
except ImportError:
    raise ImportError(
        "transformers library is required. Install it with: pip install transformers"
    )


def initialize_summarizer(model_name: str = "facebook/bart-large-cnn"):
    """
    Initialize the summarization pipeline with CPU mode.

    Args:
        model_name: Name of the pretrained model to use.

    Returns:
        A summarization pipeline configured for CPU execution.
    """
    # Force CPU mode for compatibility with GitHub runners
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=-1,  # -1 forces CPU mode
        )
        return summarizer
    except Exception as e:
        raise RuntimeError(f"Failed to initialize summarizer: {str(e)}")


def summarize_text(
    text: str,
    summarizer,
    min_length: int = 30,
    max_length: int = 150,
) -> Optional[str]:
    """
    Summarize the provided text.

    Args:
        text: The text to summarize.
        summarizer: The summarization pipeline.
        min_length: Minimum length of the summary.
        max_length: Maximum length of the summary.

    Returns:
        The summarized text or None if summarization fails.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")

    if len(text.split()) < 50:
        return "Text is too short to summarize effectively."

    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length)
        return summary[0]["summary_text"]
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}")


def main():
    """Main function to demonstrate text summarization."""
    sample_text = (
        "The quick brown fox jumps over the lazy dog. This is a sample text "
        "that demonstrates the capabilities of the text summarizer. "
        "It uses state-of-the-art transformer models from Hugging Face to "
        "generate concise and coherent summaries. The model is optimized to run "
        "on CPU, making it suitable for cloud environments and CI/CD pipelines. "
        "This approach ensures reproducibility and cost-effectiveness in automated "
        "workflows without requiring GPU resources."
    )

    print("Initializing summarizer...")
    try:
        summarizer = initialize_summarizer()
        print("✓ Summarizer initialized successfully\n")

        print("Original text:")
        print(f"{sample_text}\n")

        print("Generating summary...")
        summary = summarize_text(sample_text, summarizer)
        print("Summary:")
        print(f"{summary}\n")

        print("✓ Summarization completed successfully")

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
