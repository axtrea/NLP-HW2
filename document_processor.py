import PyPDF2
import os
from typing import List, Dict, Tuple
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch


class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50, model_name='intfloat/multilingual-e5-large'):
        """
        Initialize document processor with E5 embedding model.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            model_name: Name of the sentence transformer model
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name

        # Load the embedding model once (cached)
        self.model = None
        self.embedding_dimension = 1024  # E5-large produces 1024-dim vectors

        # For device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def _load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Model loaded successfully")
        return self.model

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""

    def split_text(self, text: str, metadata: Dict) -> Tuple[List[str], List[Dict]]:
        """Split text into coherent chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)
        metadatas = [metadata.copy() for _ in range(len(chunks))]

        # Add chunk position to metadata
        for i, meta in enumerate(metadatas):
            meta['chunk_index'] = i

        return chunks, metadatas

    def get_embeddings(self, texts: List[str], prefix: str = "passage: ") -> np.ndarray:
        """
        Get embeddings for a batch of texts using E5 model.

        Args:
            texts: List of texts to embed
            prefix: Prefix to add to each text ("query: " or "passage: ")

        Returns:
            numpy array of embeddings with shape (n_texts, 1024)
        """
        # Add the required prefix for E5 model
        prefixed_texts = [f"{prefix}{text}" for text in texts]

        # Load model if not already loaded
        model = self._load_model()

        # Get embeddings
        embeddings = model.encode(
            prefixed_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )

        return embeddings.astype(np.float32)

    def process_documents(self, file_paths: List[str]) -> Tuple[List[str], List[Dict]]:
        """Process multiple documents into chunks"""
        all_chunks = []
        all_metadatas = []

        for file_path in file_paths:
            # Extract text based on file type
            if file_path.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.txt'):
                text = self.extract_text_from_txt(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue

            if not text.strip():
                print(f"No text extracted from: {file_path}")
                continue

            # Create metadata
            metadata = {
                'source': os.path.basename(file_path),
                'file_path': file_path,
                'total_length': len(text)
            }

            # Split text
            chunks, metadatas = self.split_text(text, metadata)

            all_chunks.extend(chunks)
            all_metadatas.extend(metadatas)

            print(f"Processed {file_path}: {len(chunks)} chunks")

        return all_chunks, all_metadatas

    def create_vector_store(self, chunks: List[str], batch_size: int = 32) -> Tuple[faiss.Index, np.ndarray]:
        """
        Create FAISS vector store from chunks using E5 embeddings.

        Args:
            chunks: List of text chunks
            batch_size: Batch size for embedding generation

        Returns:
            tuple: (FAISS index, embeddings array)
        """
        print(f"Creating embeddings for {len(chunks)} chunks...")

        # Get embeddings in batches to manage memory
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"  Processing batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} "
                  f"({len(batch)} chunks)")

            batch_embeddings = self.get_embeddings(batch, prefix="passage: ")
            all_embeddings.append(batch_embeddings)

        # Combine all embeddings
        embeddings_np = np.vstack(all_embeddings)

        # Create FAISS index
        dimension = embeddings_np.shape[1]
        print(f"Creating FAISS index with dimension {dimension}...")

        index = faiss.IndexFlatL2(dimension)  # L2 distance index
        index.add(embeddings_np)

        print(f"✅ Created FAISS index with {len(chunks)} vectors of dimension {dimension}")
        return index, embeddings_np
