# KrishnaTechBD AI Automation

A production-ready AI text summarization pipeline with GitHub Actions CI/CD integration.

## Overview

This project demonstrates:
- **Text Summarization**: Uses Hugging Face Transformers (BART) for intelligent text summarization
- **CPU-Optimized**: Forces CPU mode for compatibility with GitHub runners and cloud environments
- **CI/CD Pipeline**: Automated testing, linting, and execution via GitHub Actions
- **Reproducible**: Pinned dependencies and documented workflows ensure consistent results

## Features

✓ Hugging Face Transformers integration (BART model)  
✓ CPU-only execution (no GPU required)  
✓ Comprehensive error handling and logging  
✓ Multi-version Python support (3.10, 3.11)  
✓ Flake8 linting for code quality  
✓ Pytest support for unit testing  
✓ Dependency caching for faster CI builds  

## Quick Start

### Local Setup

```bash
# Clone the repository
git clone https://github.com/KrishnaTechBD/KrishnaTechBD-ai-automation-.git
cd KrishnaTechBD-ai-automation-

# Create a virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the summarizer
python summarizer.py
```

### Usage

```python
from summarizer import initialize_summarizer, summarize_text

# Initialize the model
summarizer = initialize_summarizer()

# Summarize text
text = "Your long text here..."
summary = summarize_text(text, summarizer)
print(summary)
```

## CI/CD Automation

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:

1. **Triggers**: On every push and pull request to `main` or `develop`
2. **Matrix Testing**: Runs on Python 3.10 and 3.11
3. **Dependency Installation**: Caches pip dependencies for faster builds
4. **Linting**: Validates code with flake8
5. **Testing**: Runs pytest if `/tests` directory exists
6. **Execution**: Runs `summarizer.py` and outputs results

### Workflow Steps

```yaml
- Install Python 3.10/3.11
- Cache pip dependencies
- Install from requirements.txt
- Run flake8 linting
- Execute pytest (if tests exist)
- Run summarizer.py
```

## Project Structure

```
.
├── summarizer.py              # Main text summarization module
├── requirements.txt           # Pinned dependency versions
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions workflow
├── README.md                  # This file
└── tests/                     # Optional test directory
    └── (add test files here)
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.1.2 | PyTorch deep learning framework |
| transformers | 4.36.2 | Hugging Face Transformers library |
| datasets | 2.14.6 | Dataset utilities |
| flake8 | 6.1.0 | Code linting |
| pytest | 7.4.3 | Unit testing framework |

## Reproducibility

All dependencies are pinned to specific versions in `requirements.txt`:

```
torch==2.1.2
transformers==4.36.2
datasets==2.14.6
flake8==6.1.0
pytest==7.4.3
```

This ensures identical behavior across:
- Local development machines
- CI/CD pipelines
- Production environments

## CPU Mode Enforcement

The summarizer explicitly forces CPU mode for maximum compatibility:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = ""
summarizer = pipeline("summarization", device=-1)  # -1 = CPU
```

This makes it ideal for:
- GitHub Actions runners (no GPU available)
- Cost-effective cloud deployments
- Lightweight edge devices
- Reproducible CI/CD pipelines

## Adding Tests

Create a `tests/` directory and add test files:

```bash
mkdir tests
touch tests/test_summarizer.py
```

Example test:

```python
import pytest
from summarizer import summarize_text, initialize_summarizer

def test_summarize_text():
    summarizer = initialize_summarizer()
    text = "Sample text that is long enough to summarize. " * 10
    result = summarize_text(text, summarizer)
    assert result is not None
    assert isinstance(result, str)
```

The CI pipeline will automatically detect and run these tests.

## Troubleshooting

### Model Download Issues
The first run may take time to download the BART model (~1.6 GB). The model is cached locally.

### Memory Issues
If encountering memory errors, the model automatically uses CPU mode, which is more memory-efficient than GPU.

### Timeout in CI
The GitHub Actions workflow includes caching to speed up dependencies. If timeout issues occur, increase the timeout settings in `.github/workflows/ci.yml`.

## Future Enhancements

- [ ] Add support for multiple summarization models
- [ ] Implement batch summarization
- [ ] Add API endpoint for summarization service
- [ ] Docker containerization
- [ ] Performance benchmarking
- [ ] Model quantization for faster inference

## License

MIT License - See LICENSE file for details

## Contact

For issues or questions, please open a GitHub issue in the repository.

---

**Last Updated**: 2026-07-16  
**Status**: Production Ready ✓
