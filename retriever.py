import numpy as np

class Retriever:
    def __init__(self, index, documents, embedder):
        self.index = index
        self.documents = documents
        self.embedder = embedder

    def search(self, query: str, top_k: int):
        query_vec = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in indices[0]]
