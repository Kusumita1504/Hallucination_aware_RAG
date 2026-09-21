"""
PDF Loader and Chunking Module
Loads PDFs and splits them into chunks for vectorization
"""

import os
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    def __init__(self, data_dir: str = ".", chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize PDF Loader

        Args:
            data_dir: Directory containing PDFs
            chunk_size: Size of each chunk in characters (~500 tokens)
            chunk_overlap: Overlap between chunks
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.documents = []

    def load_pdfs(self) -> List[Dict]:
        """
        Load all PDFs from data directory

        Returns:
            List of document chunks with metadata
        """
        pdf_files = list(Path(self.data_dir).glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_dir}")

        print(f"Found {len(pdf_files)} PDF files")

        all_chunks = []

        for pdf_file in pdf_files:
            print(f"Loading {pdf_file.name}...")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            # Split into chunks
            chunks = self.splitter.split_documents(docs)

            # Add source metadata
            for chunk in chunks:
                chunk.metadata['source_file'] = pdf_file.name

            all_chunks.extend(chunks)
            print(f"  - Generated {len(chunks)} chunks from {pdf_file.name}")

        self.documents = all_chunks
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks

    def get_documents(self) -> List[Dict]:
        """Get loaded documents"""
        return self.documents


def load_and_chunk_pdfs(data_dir: str = ".") -> List[Dict]:
    """Convenience function to load and chunk PDFs"""
    loader = PDFLoader(data_dir)
    return loader.load_pdfs()
