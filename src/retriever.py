import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class ComplaintRetriever:
    def __init__(self, embeddings_path=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the retriever by loading embeddings and building FAISS index.
        """
        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Set default embeddings path (absolute path)
        if embeddings_path is None:
            # __file__ points to this file, go up one level to project root
            embeddings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "embeddings.parquet")

        embeddings_path = os.path.abspath(embeddings_path)

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        # Load parquet file
        df = pd.read_parquet(embeddings_path)

        self.documents = df["document"].tolist()
        self.metadata = df["metadata"].tolist()

        # Convert embeddings to float32 numpy array
        embeddings = np.vstack(df["embedding"].values).astype("float32")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def embed_query(self, query):
        """
        Embeds a query string using the same embedding model.
        """
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k most similar complaint chunks for a given query.
        Returns a list of dicts with keys: 'document', 'metadata', 'score'
        """
        query_embedding = self.embed_query(query)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": float(score)
            })

        return results

# ------------------ Quick test ------------------
if __name__ == "__main__":
    retriever = ComplaintRetriever()
    query = "Why are customers complaining about credit card billing issues?"
    results = retriever.retrieve(query, top_k=3)

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print("Score:", r["score"])
        print("Company:", r["metadata"]["company"])
        print("Issue:", r["metadata"]["issue"])
        print("Text:", r["document"][:200], "...\n")
