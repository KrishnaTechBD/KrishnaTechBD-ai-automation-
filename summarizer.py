from transformers import pipeline

# Load AI summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example input text
text = """
Artificial Intelligence is transforming industries by automating repetitive tasks,
improving decision-making, and enabling new innovations. However, building trustworthy
AI systems requires careful design, reproducibility, and transparency.
"""

# Run summarization
summary = summarizer(text, max_length=40, min_length=10, do_sample=False)

print("Original Text:\n", text)
print("\nAI Summary:\n", summary[0]['summary_text'])
