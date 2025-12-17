import numpy as np
import faiss
import requests
import json
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import torch


class RAGPipeline:
    def __init__(self, index: faiss.Index, chunks: List[str], metadatas: List[Dict],
                 model_name='intfloat/multilingual-e5-large'):
        """
        Initialize RAG pipeline with E5 embedding model.

        Args:
            index: FAISS index
            chunks: List of document chunks
            metadatas: Metadata for each chunk
            model_name: Name of the sentence transformer model
        """
        self.index = index
        self.chunks = chunks
        self.metadatas = metadatas
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm_model = "qwen/qwen3-30b-a3b-instruct-2507"

        # Embedding model setup
        self.model_name = model_name
        self.embedding_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_dimension = 1024

    def _load_embedding_model(self):
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            print(f"Loading query embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
        return self.embedding_model

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a query using E5 model.

        Args:
            query: User's question

        Returns:
            numpy array of embedding with shape (1, 1024)
        """
        # Add "query: " prefix as required by E5 model
        prefixed_query = f"query: {query}"

        # Load model if not already loaded
        model = self._load_embedding_model()

        # Get embedding
        embedding = model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding.astype(np.float32)

    def retrieve_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        # Get query embedding
        query_embedding = self.get_query_embedding(query)

        # Search in FAISS index (returns distances and indices)
        distances, indices = self.index.search(query_embedding, k)

        # Get retrieved chunks with metadata
        retrieved_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                # Calculate similarity score (1 - normalized distance)
                # FAISS returns L2 distances, convert to similarity score
                max_distance = 2.0  # Maximum possible distance for normalized vectors
                similarity = 1.0 - (distances[0][i] / max_distance)
                similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

                chunk_data = {
                    'text': self.chunks[idx],
                    'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {},
                    'score': similarity,
                    'distance': float(distances[0][i])
                }
                retrieved_chunks.append(chunk_data)

        # Sort by score descending
        retrieved_chunks.sort(key=lambda x: x['score'], reverse=True)

        return retrieved_chunks

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using LLM with retrieved context"""
        # Prepare context from retrieved chunks
        context = "\n\n".join([f"Document excerpt {i + 1} (Relevance: {chunk['score']:.2f}):\n{chunk['text']}"
                               for i, chunk in enumerate(context_chunks)])

        # Prepare system message
        system_message = """You are a helpful assistant that answers questions based ONLY on the provided document excerpts. 
        If the answer cannot be found in the excerpts, say "I cannot find the answer in the provided documents."
        Do not use any external knowledge. Be concise and accurate."""

        # Prepare user message with context
        user_message = f"""Question: {query}

Relevant document excerpts:
{context}

Based ONLY on the document excerpts above, please answer the question.
If the answer is not in the excerpts, say "I cannot find the answer in the provided documents."

Answer:"""

        # Call Open Router API for generation
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()

            return data['choices'][0]['message']['content'].strip()

        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

    def query(self, query: str, k: int = 3) -> Tuple[str, List[Dict]]:
        """Complete RAG query pipeline"""
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_chunks(query, k)

        if not retrieved_chunks:
            return "No relevant information found in the documents.", []

        # Generate answer
        answer = self.generate_answer(query, retrieved_chunks)

        return answer, retrieved_chunks
