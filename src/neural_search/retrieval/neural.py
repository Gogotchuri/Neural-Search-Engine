"""Neural retriever stub — implemented in Week 2 once Person C's encoder is ready."""

from typing import List, Dict

from neural_search.retrieval.base import Retriever


class NeuralRetriever(Retriever):
    """
    Wraps Person C's encoder + a FAISS IndexFlatIP over book chunks.
    Shares the same Retriever interface as BM25Retriever.
    """

    def __init__(self, index_path: str, chunks_path: str, encoder):
        raise NotImplementedError("NeuralRetriever arrives in Week 2")

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        raise NotImplementedError
