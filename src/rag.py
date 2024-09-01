import dataclasses
import numpy as np


@dataclasses.dataclass
class Database:
    texts: list
    embeddings: np.ndarray


@dataclasses.dataclass
class SimpleRAG:
    database: Database
    embedding_model: str
    prompt: str = """"""
    top_k: int = 1

    def answer_query(self, query: str, top_k=None, llm: str = 'GigaChat', model_kwargs={}, verbose=False):
        raise NotImplementedError

    async def answer_queries_a(self, queries, top_k=None, llm: str = 'GigaChat', model_kwargs={}, verbose=False,
                               return_contexts=False):
        raise NotImplementedError

    def return_context_and_prompt(self, query, top_k=None):
        raise NotImplementedError

    def query_llm(self, model, prompt, model_kwargs):
        raise NotImplementedError

    async def query_llm_a(self, model, prompt, model_kwargs):
        raise NotImplementedError

    def retrieve_context(self, query, top_k, embedding_model):
        raise NotImplementedError
