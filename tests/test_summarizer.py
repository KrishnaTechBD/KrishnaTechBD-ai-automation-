import pytest
from summarizer import summarize_text

class FakeSummarizer:
    def __call__(self, text, max_length=None, min_length=None, do_sample=False):
        return [{"summary_text": "Fake summary for testing."}]


def test_summarize_text_returns_summary():
    fake = FakeSummarizer()
    summary = summarize_text(fake, "This is a long text to summarize.", max_length=50, min_length=10)
    assert "Fake summary for testing." in summary


def test_summarize_text_empty_raises():
    fake = FakeSummarizer()
    with pytest.raises(ValueError):
        summarize_text(fake, "   ")
