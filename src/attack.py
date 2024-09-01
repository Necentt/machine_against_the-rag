from dataclasses import dataclass
import src.rag as rag

@dataclass
class NaiveAttack:
    jamming: str
    def generate_retrieval(self, target) -> str:
        raise NotImplementedError
    def generate_jamming(self) -> str:
        raise NotImplementedError
    def generate_malicious_document(self, database: rag.Database, target: str, giga_tok: str) -> rag.Database:
        raise NotImplementedError