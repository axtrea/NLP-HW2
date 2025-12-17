import streamlit as st
import os
import tempfile
import pickle
import faiss
# Import our modules
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline


class RAGSystem:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.rag = None
        self.vector_store_path = "faiss_index"
        self.metadata_path = "metadata.pkl"

    def save_vector_store(self, index, chunks, metadatas):
        """Save FAISS index and metadata"""
        faiss.write_index(index, f"{self.vector_store_path}.index")
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)

    def load_vector_store(self):
        """Load FAISS index and metadata if they exist"""
        if os.path.exists(f"{self.vector_store_path}.index") and os.path.exists(self.metadata_path):
            index = faiss.read_index(f"{self.vector_store_path}.index")
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
            return index, data["chunks"], data["metadatas"]
        return None, None, None

    def process_documents(self, uploaded_files):
        """Process uploaded documents"""
        with st.spinner("Processing documents..."):
            # Create temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(temp_path)

                # Process documents
                chunks, metadatas = self.processor.process_documents(file_paths)

                if not chunks:
                    st.error("No text could be extracted from the documents.")
                    return False

                # Create embeddings and vector store
                index, all_embeddings = self.processor.create_vector_store(chunks)

                # Save everything
                self.save_vector_store(index, chunks, metadatas)

                # Initialize RAG pipeline
                self.rag = RAGPipeline(index, chunks, metadatas)

                st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
                st.info(f"Created {len(chunks)} chunks for retrieval.")
                return True

    def answer_question(self, question):
        """Answer a question using RAG"""
        if not self.rag:
            # Try to load existing vector store
            index, chunks, metadatas = self.load_vector_store()
            if index is not None:
                self.rag = RAGPipeline(index, chunks, metadatas)
            else:
                return None, []

        with st.spinner("Searching documents and generating answer..."):
            answer, retrieved_chunks = self.rag.query(question)
            return answer, retrieved_chunks


def main():
    st.set_page_config(
        page_title="RAG Document Q&A System",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 RAG Document Q&A System")
    st.markdown("Upload documents, process them, and ask questions based on their content.")

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()

    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Processing")

        # Check for existing processed documents
        if os.path.exists("faiss_index.index"):
            st.success("📁 Found existing processed documents!")
            if st.button("Load Existing Documents"):
                index, chunks, metadatas = st.session_state.rag_system.load_vector_store()
                if index is not None:
                    st.session_state.rag_system.rag = RAGPipeline(index, chunks, metadatas)
                    st.session_state.processed = True
                    st.rerun()

        # File upload
        uploaded_files = st.file_uploader(
            "Choose documents (PDF, TXT)",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )

        # Process button
        if uploaded_files and st.button("Process Documents", type="primary"):
            success = st.session_state.rag_system.process_documents(uploaded_files)
            if success:
                st.session_state.processed = True
                st.rerun()

        st.markdown("---")
        st.markdown("### How to Use:")
        st.markdown("""
        1. Upload PDF or TXT documents
        2. Click 'Process Documents'
        3. Wait for processing to complete
        4. Ask questions in the main panel
        """)

    # Main content area
    if st.session_state.processed:
        st.success("✅ Documents are ready for questioning!")

        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main findings?",
            key="question_input"
        )

        if question:
            # Get answer
            answer, retrieved_chunks = st.session_state.rag_system.answer_question(question)

            if answer:
                # Display answer
                st.subheader("Answer")
                st.markdown(f"**{answer}**")

                # Display retrieved chunks
                st.subheader("Retrieved Chunks Used")
                for i, chunk in enumerate(retrieved_chunks[:3], 1):  # Show top 3 chunks
                    with st.expander(f"Chunk {i} (Relevance: {chunk.get('score', 'N/A'):.3f})"):
                        st.markdown(chunk['text'])
                        if chunk.get('metadata'):
                            st.caption(f"Source: {chunk['metadata'].get('source', 'Unknown')}")
            else:
                st.error("Failed to generate answer. Please try again.")
    else:
        # Welcome message when no documents processed
        st.info("👈 Please upload and process documents in the sidebar to begin.")

        # Example documents section
        with st.expander("📋 Example Document Types (5-10 pages)"):
            st.markdown("""
            **Good document types for this system:**
            - Research papers or reports
            - Technical manuals
            - Business plans
            - Legal documents
            - Academic articles
            - Product specifications

            **Minimum size:** 5-10 pages of text
            **Formats supported:** PDF, TXT
            """)


if __name__ == "__main__":
    # Check for Open Router API key
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("Please set your Open Router API key as an environment variable: OPENROUTER_API_KEY")
        st.info("Get your API key from: https://openrouter.ai/keys")
    else:
        main()
