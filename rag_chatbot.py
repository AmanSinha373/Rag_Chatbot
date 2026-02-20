from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RetrievalResult:
    chunk: str
    score: float


class RAGChatbot:
    """Simple local RAG chatbot using pure-Python TF-IDF retrieval."""

    def __init__(self, knowledge_base_path: str | Path, chunk_size: int = 500, overlap: int = 80):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[str] = []
        self.chunk_vectors: List[Counter[str]] = []
        self.idf: dict[str, float] = {}

    def load_and_index(self) -> None:
        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {self.knowledge_base_path}")

        raw_text = self.knowledge_base_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            raise ValueError("Knowledge base is empty. Add text and retry.")

        self.chunks = self._chunk_text(raw_text)
        tokenized_chunks = [self._tokenize(c) for c in self.chunks]

        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_chunks:
            doc_freq.update(set(tokens))

        total_docs = len(tokenized_chunks)
        self.idf = {term: math.log((1 + total_docs) / (1 + freq)) + 1 for term, freq in doc_freq.items()}

        self.chunk_vectors = [self._tfidf_vector(tokens) for tokens in tokenized_chunks]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _tfidf_vector(self, tokens: List[str]) -> Counter[str]:
        tf = Counter(tokens)
        length = len(tokens) or 1
        vector: Counter[str] = Counter()
        for term, count in tf.items():
            vector[term] = (count / length) * self.idf.get(term, 0.0)
        return vector

    def _cosine_similarity(self, a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        common = set(a.keys()) & set(b.keys())
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _chunk_text(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - self.overlap
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        if not self.chunk_vectors or not self.chunks:
            raise RuntimeError("Index not initialized. Call load_and_index() first.")

        query_tokens = self._tokenize(query)
        query_vector = self._tfidf_vector(query_tokens)

        scores = [self._cosine_similarity(query_vector, chunk_vec) for chunk_vec in self.chunk_vectors]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [RetrievalResult(chunk=self.chunks[i], score=score) for i, score in ranked if score > 0]

    def answer(self, query: str, top_k: int = 3, min_score: float = 0.08) -> str:
        results = self.retrieve(query, top_k=top_k)
        strong_results = [r for r in results if r.score >= min_score]

        if not strong_results:
            return (
                "I don't have enough relevant context in the current knowledge base to answer that confidently. "
                "Please add more source content to data/knowledge_base.txt and ask again."
            )

        context = "\n\n".join(
            f"[{idx + 1}] score={res.score:.3f}\n{res.chunk}" for idx, res in enumerate(strong_results)
        )

        return (
            "Answer (grounded in retrieved knowledge):\n"
            f"Based on the most relevant context, here is what I found:\n\n{context}\n\n"
            "If you want a better or broader answer, expand the knowledge base with more domain documents."
        )
