import faiss
import pickle
import numpy as np

class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def save(self, index_path: str, docs_path: str, documents: list):
        faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(documents, f)

    @staticmethod
    def load(index_path: str, docs_path: str):
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)
        return index, documents
