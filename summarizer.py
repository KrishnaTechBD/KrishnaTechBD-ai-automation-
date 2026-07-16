#!/usr/bin/env python3
"""
summarizer.py - simple CLI summarizer using HuggingFace transformers

Usage examples:
  python summarizer.py                       # uses built-in example text
  python summarizer.py --text "Some long text to summarize..."
  python summarizer.py --file path/to/file.txt
"""

import argparse
import textwrap
import torch
from transformers import pipeline


def build_summarizer(model_name: str = "facebook/bart-large-cnn"):
    # Use GPU if available, otherwise CPU (-1)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, device=device)


def summarize_text(summarizer, text: str, max_length: int = 130, min_length: int = 30) -> str:
    if not text or not text.strip():
        raise ValueError("Input text is empty")
    # The pipeline returns a list of dicts; we extract the summary_text
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0].get("summary_text", "").strip()


def main():
    parser = argparse.ArgumentParser(description="Summarize text with HuggingFace BART.")
    parser.add_argument("--file", "-f", help="Path to a text file to summarize")
    parser.add_argument("--text", "-t", help="Text to summarize (use quotes).")
    parser.add_argument("--max_length", type=int, default=130, help="Maximum summary length")
    parser.add_argument("--min_length", type=int, default=30, help="Minimum summary length")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
    elif args.text:
        text = args.text
    else:
        text = textwrap.dedent("""
            Artificial Intelligence is transforming industries by automating repetitive tasks,
            improving decision-making, and enabling new innovations. However, building trustworthy
            AI systems requires careful design, reproducibility, and transparency.
        """).strip()

    summarizer = build_summarizer()
    try:
        summary = summarize_text(summarizer, text, max_length=args.max_length, min_length=args.min_length)
    except Exception as e:
        raise SystemExit(f"Summarization failed: {e}")

    print("Original Text:\n", text)
    print("\nAI Summary:\n", summary)


if __name__ == "__main__":
    main()
