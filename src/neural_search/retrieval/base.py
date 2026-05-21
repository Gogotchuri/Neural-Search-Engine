from abc import ABC, abstractmethod
from typing import List, Dict


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Return top-k passages for a query.

        Each result dict contains: chunk_id, text, score, chapter, section.
        """
        ...
