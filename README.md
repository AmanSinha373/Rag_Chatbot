# RAG Chatbot

A lightweight Retrieval-Augmented Generation (RAG) chatbot that answers questions using a local knowledge base.

## What this project does

- Loads `data/knowledge_base.txt`.
- Splits it into overlapping chunks.
- Builds a TF-IDF retriever.
- Retrieves the most relevant chunks for each query.
- Returns an answer grounded only in retrieved chunks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Add your own knowledge

Edit `data/knowledge_base.txt` with your domain content, then restart the app.

The chatbot quality depends on your source text. More relevant knowledge gives better answers.
