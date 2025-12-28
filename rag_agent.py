import os
import pickle
import faiss
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message
import numpy as np


@dataclass
class Chunk:
    """Represents a retrieved chunk with metadata"""
    text: str
    doc_id: str
    page: Optional[int]
    chunk_id: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'doc_id': self.doc_id,
            'page': self.page,
            'chunk_id': self.chunk_id,
            'score': self.score
        }


class CustomLLM(BaseChatModel):
    """Custom LLM wrapper for OpenRouter API with all required methods implemented"""

    def __init__(self, model_name: str = "qwen/qwen3-30b-a3b-instruct-2507"):
        super().__init__()
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

    def _chat_no_stream(self, messages: List[Message], **kwargs) -> Message:
        """Non-streaming chat implementation"""
        return self._call_openrouter_api(messages, stream=False, **kwargs)

    def _chat_stream(self, messages: List[Message], **kwargs) -> Iterator[Message]:
        """Streaming chat implementation - simplified for OpenRouter"""
        # For simplicity, we'll just call non-streaming and yield once
        result = self._call_openrouter_api(messages, stream=False, **kwargs)
        yield result

    def _chat_with_functions(self, messages: List[Message], functions: List[Dict], **kwargs) -> Message:
        """Chat with function calling - simplified implementation"""
        # Add functions to the API call
        kwargs['functions'] = functions
        return self._call_openrouter_api(messages, stream=False, **kwargs)

    def _call_openrouter_api(self, messages: List[Message], stream: bool = False, **kwargs) -> Message:
        """Call OpenRouter API"""
        import requests

        # Convert messages to OpenRouter format
        formatted_messages = []
        for msg in messages:
            # Handle both string and ContentItem messages
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Extract text from ContentItem list
                content = ""
                for item in msg.content:
                    if isinstance(item, dict) and 'text' in item:
                        content += item['text'] + "\n"
                    elif isinstance(item, str):
                        content += item + "\n"
            else:
                content = str(msg.content)

            formatted_messages.append({
                'role': msg.role,
                'content': content.strip()
            })

        # Call OpenRouter API
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": 0.1,
            "max_tokens": 1000
        }

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in payload and key not in ['functions', 'function_call']:
                payload[key] = value

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            return Message(
                role='assistant',
                content=result['choices'][0]['message']['content']
            )
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return Message(
                role='assistant',
                content=f"Error generating response: {str(e)}"
            )


class RAGRetriever:
    """Handles retrieval from FAISS index"""

    def __init__(self, index_path: str = "faiss_index.index",
                 metadata_path: str = "metadata.pkl",
                 embedding_model: str = 'intfloat/multilingual-e5-large'):

        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load index and metadata
        self.index, self.chunks, self.metadatas = self._load_vector_store()
        if self.index is None:
            raise ValueError(f"Could not load vector store from {index_path}")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device=self.device
        )

        # Build document index for stats
        self._build_document_index()

    def _load_vector_store(self):
        """Load FAISS index and metadata"""
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            return None, None, None

        index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)

        return index, data.get("chunks", []), data.get("metadatas", [])

    def _build_document_index(self):
        """Build index of documents for stats"""
        self.documents = {}
        for i, metadata in enumerate(self.metadatas):
            doc_id = metadata.get('source', f'doc_{i}')
            if doc_id not in self.documents:
                self.documents[doc_id] = {
                    'chunk_count': 0,
                    'pages': set()
                }
            self.documents[doc_id]['chunk_count'] += 1

            # Extract page number if available
            page = metadata.get('page', metadata.get('chunk_index', 0))
            self.documents[doc_id]['pages'].add(page)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query using E5 model"""
        prefixed_query = f"query: {query}"
        embedding = self.embedding_model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.astype(np.float32)

    def rag_search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Chunk]:
        """Search for relevant chunks"""
        # Get query embedding
        query_embedding = self.get_query_embedding(query)

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Convert to Chunk objects
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                # Calculate similarity score (1 - normalized distance)
                max_distance = 2.0
                similarity = 1.0 - (distances[0][i] / max_distance)
                similarity = max(0.0, min(1.0, similarity))

                metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}
                doc_id = metadata.get('source', f'doc_{idx}')
                page = metadata.get('page', metadata.get('chunk_index', 0))
                chunk_id = f"{doc_id}_c{metadata.get('chunk_index', idx)}"

                chunk = Chunk(
                    text=self.chunks[idx],
                    doc_id=doc_id,
                    page=page,
                    chunk_id=chunk_id,
                    score=similarity
                )

                # Apply filters if provided
                if filters:
                    if 'doc_id' in filters and doc_id != filters['doc_id']:
                        continue

                results.append(chunk)

        return results

    def kb_status(self) -> Dict[str, Any]:
        """Check KB status"""
        status = {
            'ready': self.index is not None,
            'path': os.path.abspath(self.index_path),
            'index_type': 'FAISS',
            'embedding_model': self.embedding_model_name
        }

        if self.index is None:
            status['error'] = 'Index not loaded'

        return status

    def kb_stats(self) -> Dict[str, Any]:
        """Get KB statistics"""
        if not self.index:
            return {'error': 'Index not loaded'}

        total_chunks = len(self.chunks)
        total_docs = len(self.documents)
        embed_dim = self.index.d

        # Calculate average chunks per document
        avg_chunks_per_doc = total_chunks / total_docs if total_docs > 0 else 0

        # Get unique pages
        all_pages = set()
        for doc_info in self.documents.values():
            all_pages.update(doc_info['pages'])

        stats = {
            'documents': total_docs,
            'chunks': total_chunks,
            'embed_dim': embed_dim,
            'index_type': 'FAISS',
            'avg_chunks_per_doc': round(avg_chunks_per_doc, 2),
            'unique_pages': len(all_pages),
            'documents_list': list(self.documents.keys())[:10]  # First 10 docs
        }

        return stats

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get specific chunk by ID (extra tool)"""
        # Parse chunk_id format: {doc_id}_c{chunk_index}
        try:
            if '_c' in chunk_id:
                doc_part, chunk_part = chunk_id.rsplit('_c', 1)
                chunk_idx = int(chunk_part)

                # Find chunk by index in metadata
                for i, metadata in enumerate(self.metadatas):
                    current_doc = metadata.get('source', f'doc_{i}')
                    current_idx = metadata.get('chunk_index', i)

                    if current_doc == doc_part and current_idx == chunk_idx:
                        return Chunk(
                            text=self.chunks[i],
                            doc_id=current_doc,
                            page=metadata.get('page', metadata.get('chunk_index', 0)),
                            chunk_id=chunk_id,
                            score=1.0
                        )
        except (ValueError, IndexError):
            pass

        return None


class ToolTracer:
    """Tracks and prints tool usage"""

    def __init__(self):
        self.tool_calls = []

    def add_call(self, tool_name: str, params: Dict, result: Any):
        """Add a tool call to trace"""
        # Format result for display
        if isinstance(result, list):
            if result and hasattr(result[0], 'chunk_id'):
                # Format chunks for display
                formatted = [f"{chunk.chunk_id} ({chunk.score:.2f})"
                             for chunk in result[:3]]
                if len(result) > 3:
                    formatted.append(f"... and {len(result) - 3} more")
                result_str = formatted
            else:
                result_str = f"list[{len(result)} items]"
        elif isinstance(result, dict):
            # Show only key names for large dicts
            result_str = f"dict with keys: {list(result.keys())}"
        else:
            result_str = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)

        self.tool_calls.append({
            'tool': tool_name,
            'params': params,
            'result': result_str
        })

    def print_trace(self):
        """Print the tool trace"""
        if not self.tool_calls:
            return

        print("\n" + "=" * 60)
        print("TOOL TRACE:")
        print("=" * 60)

        for call in self.tool_calls:
            print(f"\n{call['tool']}({call['params']})")
            print(f"  -> {call['result']}")

        print("=" * 60 + "\n")

        # Clear for next interaction
        self.tool_calls = []


class RAGAgent:
    """Main agent that coordinates RAG and tools"""

    def __init__(self):
        # Initialize components
        self.retriever = RAGRetriever()
        self.tracer = ToolTracer()

        # Initialize LLM with required abstract methods implemented
        self.llm = CustomLLM()

        # System prompt for the agent
        self.system_prompt = """You are a helpful document assistant with access to a knowledge base (KB).

RULES:
1. For document questions: ALWAYS use rag_search first. Answer ONLY from retrieved chunks.
2. If user asks about KB contents: Use kb_status or kb_stats as appropriate.
3. If rag_search doesn't find relevant information: Say "I cannot find the answer in the knowledge base."
4. NEVER guess or use external knowledge for document questions.
5. For document-based answers: Include an "Evidence/Sources" section listing sources used.

TOOLS AVAILABLE:
- rag_search(query, top_k=5): Search the KB for relevant chunks
- kb_status(): Check if KB is available and get basic info
- kb_stats(): Get statistics about the KB contents
- get_chunk_by_id(chunk_id): Get a specific chunk by its ID

ALWAYS think step by step and use the appropriate tools."""

    def process_query(self, query: str) -> str:
        """Process a user query"""
        print(f"\n📝 Query: {query}")

        # Reset tracer for new query
        self.tracer = ToolTracer()

        # Check if it's a KB status question
        kb_keywords = ['status', 'available', 'loaded', 'what is in', 'contents', 'statistics', 'stats']
        is_kb_query = any(keyword in query.lower() for keyword in kb_keywords)

        if is_kb_query:
            # Use KB tools
            if 'stat' in query.lower():
                result = self.retriever.kb_stats()
                self.tracer.add_call('kb_stats', {}, result)
            else:
                result = self.retriever.kb_status()
                self.tracer.add_call('kb_status', {}, result)

            # Format response
            response = self._format_kb_response(result, query)

        else:
            # Document question - use RAG
            chunks = self.retriever.rag_search(query, top_k=5)
            self.tracer.add_call('rag_search', {'query': query, 'top_k': 5}, chunks)

            if not chunks:
                response = "I cannot find the answer in the knowledge base."
            else:
                # Generate answer using LLM
                response = self._generate_answer(query, chunks)

                # Add evidence section
                response += self._format_evidence(chunks)

        # Print tool trace
        self.tracer.print_trace()

        return response

    def _format_kb_response(self, kb_info: Dict, query: str) -> str:
        """Format KB information response"""
        if 'error' in kb_info:
            return f"KB Status: {kb_info['error']}"

        if 'stat' in query.lower():
            # Format stats
            response = "📊 Knowledge Base Statistics:\n\n"
            response += f"• Documents: {kb_info.get('documents', 'N/A')}\n"
            response += f"• Total chunks: {kb_info.get('chunks', 'N/A')}\n"
            response += f"• Embedding dimension: {kb_info.get('embed_dim', 'N/A')}\n"
            response += f"• Index type: {kb_info.get('index_type', 'N/A')}\n"
            response += f"• Average chunks per document: {kb_info.get('avg_chunks_per_doc', 'N/A')}\n"

            if 'documents_list' in kb_info:
                response += f"\n• Sample documents: {', '.join(kb_info['documents_list'])}"
                if kb_info.get('documents', 0) > 10:
                    response += f" ... and {kb_info['documents'] - 10} more"

        else:
            # Format status
            response = "✅ Knowledge Base Status:\n\n"
            response += f"• Ready: {kb_info.get('ready', False)}\n"
            response += f"• Path: {kb_info.get('path', 'N/A')}\n"
            response += f"• Index type: {kb_info.get('index_type', 'N/A')}\n"
            response += f"• Embedding model: {kb_info.get('embedding_model', 'N/A')}"

        return response

    def _generate_answer(self, query: str, chunks: List[Chunk]) -> str:
        """Generate answer using LLM with retrieved context"""
        # Prepare context
        context = "\n\n".join([
            f"[Chunk {i + 1} - Score: {chunk.score:.3f}]\n{chunk.text}"
            for i, chunk in enumerate(chunks[:3])  # Use top 3 chunks
        ])

        # Prepare messages for LLM
        messages = [
            Message(role='system', content=self.system_prompt),
            Message(role='user', content=f"""Question: {query}

Relevant document excerpts:
{context}

Based ONLY on the document excerpts above, please answer the question.
If the answer cannot be found, say "I cannot find the answer in the knowledge base."

Answer concisely:""")
        ]

        # Get response from LLM
        response_msg = self.llm._chat_no_stream(messages, temperature=0.1, max_tokens=500)
        return response_msg.content

    def _format_evidence(self, chunks: List[Chunk]) -> str:
        """Format evidence/sources section"""
        if not chunks:
            return ""

        evidence = "\n\n---\n**Evidence / Sources:**\n\n"

        # Group by document
        doc_groups = {}
        for chunk in chunks[:5]:  # Show top 5 sources
            if chunk.doc_id not in doc_groups:
                doc_groups[chunk.doc_id] = []
            doc_groups[chunk.doc_id].append(chunk)

        for doc_id, doc_chunks in doc_groups.items():
            evidence += f"• **{doc_id}**:\n"
            for chunk in doc_chunks:
                location = f"page {chunk.page}" if chunk.page else f"chunk {chunk.chunk_id}"
                evidence += f"  - {location} (relevance: {chunk.score:.3f})\n"

        return evidence

    def run_console(self):
        """Run the agent in console mode"""
        print("\n" + "=" * 60)
        print("RAG AGENT CONSOLE")
        print("=" * 60)
        print("\nKnowledge Base loaded and ready!")
        print("Type 'quit' or 'exit' to end the session.\n")

        while True:
            try:
                # Get user input
                query = input("\n💭 Your question: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break

                if not query:
                    continue

                # Process query
                response = self.process_query(query)

                # Print response
                print("\n" + "=" * 60)
                print("🤖 AGENT RESPONSE:")
                print("=" * 60)
                print(response)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point"""
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Get your API key from: https://openrouter.ai/keys")
        print("Then set it with: export OPENROUTER_API_KEY='your-key-here'")
        return

    try:
        # Initialize and run agent
        agent = RAGAgent()
        agent.run_console()

    except ValueError as e:
        print(f"❌ Error: {str(e)}")
        print("\nMake sure you have processed documents first and have:")
        print("1. faiss_index.index")
        print("2. metadata.pkl")
        print("\nRun your Streamlit app first to process documents.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
