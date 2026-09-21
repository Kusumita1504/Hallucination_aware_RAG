"""
Vector Store Module
Manages FAISS index for efficient similarity search
"""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle


class FAISSVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_db_path: str = "vector_db"):
        """
        Initialize FAISS Vector Store

        Args:
            model_name: Sentence-transformer model to use
            vector_db_path: Path to store/load FAISS index
        """
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dim = 384 if "L6" in model_name else 768  # embedding dimension
        self.source_signature = None

    def create_index(self, documents: List, source_signature: Optional[str] = None) -> None:
        """
        Create FAISS index from documents

        Args:
            documents: List of langchain documents with page_content
        """
        print(f"Creating embeddings for {len(documents)} documents...")

        # Extract text from documents
        texts = [doc.page_content for doc in documents]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)

        self.documents = documents
        self.source_signature = source_signature
        print(f"Index created with {len(documents)} documents")

        # Save index
        self._save_index()

    def _save_index(self) -> None:
        """Save FAISS index to disk"""
        os.makedirs(self.vector_db_path, exist_ok=True)
        index_path = os.path.join(self.vector_db_path, "faiss_index.bin")
        docs_path = os.path.join(self.vector_db_path, "documents.pkl")
        faiss.write_index(self.index, index_path)
        print(f"Index saved to {index_path}")
        with open(docs_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "source_signature": self.source_signature,
                    "model_name": self.model_name,
                },
                f,
            )

    def load_index(self, expected_signature: Optional[str] = None) -> bool:
        """Load FAISS index and cached documents from disk if present and fresh."""
        index_path = os.path.join(self.vector_db_path, "faiss_index.bin")
        docs_path = os.path.join(self.vector_db_path, "documents.pkl")
        if not (os.path.exists(index_path) and os.path.exists(docs_path)):
            return False

        try:
            with open(docs_path, "rb") as f:
                cache = pickle.load(f)
            cached_signature = cache.get("source_signature")

            if expected_signature is not None and cached_signature != expected_signature:
                print("Cached index is stale. Rebuilding...")
                return False

            self.index = faiss.read_index(index_path)
            self.documents = cache.get("documents", [])
            self.source_signature = cached_signature

            if not self.documents:
                print("Cached index missing documents. Rebuilding...")
                return False

            print(f"Index loaded from {index_path}")
            return True
        except Exception:
            return False

    def search(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for similar documents

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not initialized. Create index first.")

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:  # Valid index
                results.append((self.documents[idx], float(distance)))

        return results

    def retrieve_top_k(self, query: str, k: int = 5) -> List:
        """Retrieve top-k documents for query"""
        results = self.search(query, k)
        return [doc for doc, _ in results]
