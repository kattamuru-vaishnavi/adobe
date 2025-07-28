# 🧠 Challenge 1B – Context-Aware PDF Section Extractor

This project extracts and ranks the most relevant sections from a set of PDF documents based on a **user persona** and a **job-to-be-done (task)**.  
It uses a **pretrained transformer model** to embed both the task and the document content, then performs **semantic similarity ranking** to return the top-matching content chunks.

---

## 📁 Folder Structure

challenge1b/
├── Dockerfile
├── requirements.txt
├── analyser.py
├── input/
│   └── collection1/
│       ├── challenge1b_input.json
│       └── pdfs/
│           ├── file1.pdf
│           ├── file2.pdf
│           └── ...
├── output/
└── distiluse-model/
    ├── config.json
    ├── sentence_bert_config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.txt
---

## 🔧 Setup Instructions

### 1. Install Dependencies

Ensure Python 3.8+ is installed.  
Install the required packages using:

```bash
pip install -r requirements.txt

```
### 2. Download the Embedding Model

To download the required embedding model (`distiluse-base-multilingual-cased-v1`), simply run:

```bash
python download_model.py
