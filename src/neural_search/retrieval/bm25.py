import json
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi

from neural_search.retrieval.base import Retriever


class BM25Retriever(Retriever):
    def __init__(self, chunks_path: str):
        self._chunks = self._load(chunks_path)
        tokenized_chunks = [chunk["text"].lower().split() for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_chunks)

    @staticmethod
    def _load(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as in_file:
            return [json.loads(line) for line in in_file]

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            {
                "chunk_id": self._chunks[idx]["chunk_id"],
                "text": self._chunks[idx]["text"],
                "score": float(scores[idx]),
                "chapter": self._chunks[idx]["chapter"],
                "section": self._chunks[idx]["section"],
            }
            for idx in top_indices
        ]
