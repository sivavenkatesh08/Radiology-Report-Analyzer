# 🏥 AI Radiology Assistant

An AI-powered production-level system for analyzing radiology reports and medical images using **RAG + Agent-based reasoning**.

## ✨ Features

### Three Input Modes
1. **📄 Text Report** — Paste radiology text for instant AI analysis
2. **🧾 Report Upload** — Upload scanned reports (Image/PDF) with OCR extraction
3. **🩻 X-ray / MRI** — Upload medical scans for anomaly detection

### RAG + Agent Architecture
- **FAISS Vector Store** — 19 curated medical knowledge documents indexed with sentence-transformer embeddings
- **Semantic Retrieval** — Cosine similarity search returns top-3 relevant medical contexts
- **Agent-Based Reasoning** — Combines rule-based interpretation + RAG context into unified clinical decisions
- **Decision Fusion** — Severity escalation from RAG, augmented recommendations, transparent context display

### Analysis Capabilities
- **Transformer Summarization** — BART-Large-CNN for abstractive clinical summaries
- **Rule-Based Interpretation** — 20+ medical condition patterns with weighted severity
- **Key Findings Extraction** — Structured findings table with per-finding severity
- **OCR Pipeline** — Tesseract OCR with CLAHE preprocessing, deskewing, and noise filtering
- **Anomaly Detection** — OpenCV contour analysis with bounding boxes and region labeling
- **U-Net Architecture** — Deep learning segmentation scaffold (ready for training)
- **Downloadable Reports** — Premium HTML reports with severity color-coding and findings table

### Output (Clean Medical Only)
- Clinical Summary
- Clinical Interpretation (with RAG-augmented context)
- Key Findings Table
- RAG Knowledge Context (retrieved topics + relevance scores)
- Severity Level (Low / Medium / High)
- Clinical Recommendation
- Downloadable HTML Report

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| NLP | Transformers (BART-Large-CNN), BioBERT NER |
| RAG | FAISS + Sentence-Transformers (all-MiniLM-L6-v2) |
| Agent | Rule-based + RAG decision fusion engine |
| CV | OpenCV (prototype), U-Net PyTorch (advanced) |
| OCR | Tesseract + CLAHE + pdf2image |
| UI | Streamlit with custom CSS |

## 📁 Project Structure

```
├── app/
│   └── app.py                    # Streamlit application (3 tabs)
├── src/
│   ├── agents/
│   │   └── radiology_agent.py    # Agent: rule-based + RAG fusion
│   ├── rag/
│   │   ├── knowledge_base.py     # 19 curated medical knowledge docs
│   │   └── vector_store.py       # FAISS index + sentence-transformer
│   ├── nlp/
│   │   ├── summarizer.py         # BART summarization
│   │   ├── interpreter.py        # Medical interpretation (20+ rules)
│   │   ├── ocr.py                # OCR with preprocessing & fallback
│   │   └── ner.py                # Biomedical NER
│   ├── vision/
│   │   ├── image_processor.py    # OpenCV anomaly detection (prototype)
│   │   └── unet_model.py         # U-Net segmentation architecture
│   └── utils/
│       └── report_generator.py   # HTML report generator
├── data/
│   ├── images/                   # Sample X-rays
│   └── reports/                  # Sample reports
├── main.py                       # CLI entry point
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for report upload)
- [Poppler](https://poppler.freedesktop.org/) (for PDF processing)

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app/app.py
```

## 📝 Notes

- **RAG Pipeline**: FAISS index is built automatically on startup from 19 medical knowledge documents
- **Agent Reasoning**: Combines rule-based findings with RAG-retrieved context for comprehensive analysis
- **OCR Fallback**: If Tesseract is not installed, the app gracefully shows an error message
- **U-Net**: Architecture is fully implemented — train with NIH Chest X-ray or CheXpert datasets
- **Prototype Mode**: Uses OpenCV contour detection — may produce false positives
- **Disclaimer**: For informational purposes only. Not a substitute for professional medical diagnosis.

## 📄 License

MIT