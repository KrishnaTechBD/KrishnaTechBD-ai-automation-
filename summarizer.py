from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # force CPU
)

text = "AI automation helps reduce repetitive tasks and improves workflows."
summary = summarizer(text, max_length=40, min_length=10, do_sample=False)

print("Summary:", summary[0]['summary_text'])
