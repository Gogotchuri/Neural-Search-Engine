"""Neural retriever using a trained transformer encoder + FAISS index."""

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from neural_search.encoder.encoder import Encoder
from neural_search.retrieval.base import Retriever


class NeuralRetriever(Retriever):
    """Dense retriever over book chunks using a trained Encoder + FAISS.

    Uses FAISS IndexFlatIP (inner product) for exact nearest-neighbor search.
    Since the encoder produces L2-normalized embeddings, inner product equals
    cosine similarity.
    We can go with the exact nearest neighbor search since the chunks are
    small enough to fit in memory.
    """

    def __init__(
        self,
        index_path: str,
        chunks_path: str,
        encoder: Encoder,
        tokenizer,
    ):
        """
        Args:
            index_path:  Path to a FAISS index file (.faiss). If the file
                         exists, it is loaded; otherwise call build_index().
            chunks_path: Path to chunks.jsonl (same format as BM25Retriever).
            encoder:     A trained Encoder instance.
            tokenizer:   A HuggingFace tokenizers.Tokenizer instance.
        """
        self._index_path = index_path
        self._chunks = self._load_chunks(chunks_path)
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._index: faiss.IndexFlatIP | None = None

        # Load pre-built index if it exists
        if Path(index_path).exists():
            self._index = faiss.read_index(index_path)

    @staticmethod
    def _load_chunks(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def build_index(self, device: str = "cpu", batch_size: int = 64) -> None:
        """Encode all chunks and build the FAISS index.

        This is expensive (encodes every chunk through the transformer) and
        should be run once, then the index saved to disk.
        """
        texts = [chunk["text"] for chunk in self._chunks]

        print(f"Encoding {len(texts)} chunks...")
        embeddings = self._encoder.encode(
            texts,
            self._tokenizer,
            batch_size=batch_size,
            device=device,
        )  # (N, hidden_dim), L2-normalized float32

        # Build the FAISS inner-product index
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))

        # Persist to disk
        faiss.write_index(self._index, self._index_path)
        print(f"FAISS index saved → {self._index_path} ({self._index.ntotal} vectors)")

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        if self._index is None:
            raise RuntimeError(
                "No FAISS index loaded. Call build_index() or provide an existing index file."
            )

        # Encode the query
        query_emb = self._encoder.encode(
            [query], self._tokenizer
        )  # (1, hidden_dim)

        # Search: returns (scores, indices) each of shape (1, k)
        scores, indices = self._index.search(query_emb.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
            chunk = self._chunks[idx]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": float(score),
                    "chapter": chunk["chapter"],
                    "section": chunk["section"],
                }
            )

        return results
