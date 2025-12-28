#!/usr/bin/env python3
"""
Simple launcher for the RAG Agent console.
Make sure you have processed documents first using the Streamlit app.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_agent import main
    main()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure you have installed all required packages:")
    print("pip install qwen-agent sentence-transformers faiss-cpu langchain PyPDF2")
    sys.exit(1)
