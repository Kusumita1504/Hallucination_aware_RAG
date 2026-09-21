"""
JSON-based Retriever for Structured Disease Knowledge
Provides balanced disease coverage with metadata filtering
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from .vector_store import FAISSVectorStore


class JSONRetriever:
    def __init__(self, json_path: str, embeddings):
        """
        Initialize JSON Retriever
        
        Args:
            json_path: Path to disease_knowledge.json
            embeddings: Embeddings instance for vectorization
        """
        self.json_path = json_path
        self.embeddings = embeddings
        self.diseases = []
        self.chunks = []
        self.vector_store = None
        
        # Load and index JSON data
        self._load_json()
        self._create_chunks()
        self._build_index()
    
    def _load_json(self):
        """Load disease data from JSON file"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.diseases = data['diseases']
    
    def _create_chunks(self):
        """
        Convert JSON disease records into searchable text chunks
        Each chunk represents a disease with structured information
        """
        for disease in self.diseases:
            # Create comprehensive text representation
            symptoms_text = disease['symptoms']['description']
            pattern_text = f"Pattern: {disease['symptoms']['pattern']}"
            location_text = f"Location: {disease['symptoms']['location']}"
            color_text = f"Color: {disease['symptoms']['color']}"
            progression_text = f"Progression: {disease['symptoms']['progression']}"
            
            causes_text = "Causes: " + "; ".join(disease['causes'])
            treatment_text = "Treatment: " + "; ".join(disease['treatment'])
            prevention_text = "Prevention: " + "; ".join(disease['prevention'])
            
            # Combine into full chunk
            chunk_text = f"""
Disease: {disease['disease_name']}
Severity: {disease['severity']}

Symptoms:
{symptoms_text}
{pattern_text}
{location_text}
{color_text}
{progression_text}

{causes_text}

{treatment_text}

{prevention_text}
""".strip()
            
            # Create chunk with metadata
            chunk = {
                'content': chunk_text,
                'source': 'disease_knowledge.json',
                'source_type': 'JSON',
                'metadata': {
                    'disease_id': disease['disease_id'],
                    'disease_name': disease['disease_name'],
                    'severity': disease['severity'],
                    'region': disease['region'],
                    'symptom_type': disease['metadata']['symptom_type'],
                    'affected_parts': disease['metadata']['affected_parts'],
                    'season': disease['metadata']['season'],
                    'diagnostic_confidence': disease['metadata']['diagnostic_confidence'],
                    'source_file': 'disease_knowledge.json'
                }
            }
            
            self.chunks.append(chunk)
    
    def _build_index(self):
        """Build FAISS index for JSON chunks"""
        # Create a dedicated FAISS store for JSON chunks
        json_index_dir = str(Path(self.json_path).resolve().parent / "vector_db_json")
        self.vector_store = FAISSVectorStore(
            model_name="all-MiniLM-L6-v2",
            vector_db_path=json_index_dir
        )

        # Reuse the same embedding model object from the main PDF vector store (if provided)
        if self.embeddings is not None:
            self.vector_store.embedding_model = self.embeddings

        # Build/load index from documents with metadata
        json_signature = self._compute_json_signature()
        if self.vector_store.load_index(expected_signature=json_signature):
            return

        from langchain_core.documents import Document
        docs = []
        for chunk in self.chunks:
            doc = Document(
                page_content=chunk['content'],
                metadata=chunk['metadata']
            )
            docs.append(doc)

        self.vector_store.create_index(docs, source_signature=json_signature)

    def _compute_json_signature(self) -> str:
        """Signature from JSON file modified time + size."""
        p = Path(self.json_path)
        stat = p.stat()
        return f"{p.name}:{int(stat.st_mtime)}:{stat.st_size}"
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        symptom_type_filter: Optional[str] = None,
        region_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant disease chunks from JSON
        
        Args:
            query: User query
            top_k: Number of results to return
            symptom_type_filter: Filter by symptom type (foliar_fungal, nutritional, etc.)
            region_filter: Filter by region (Karnataka, General)
        
        Returns:
            List of chunks with metadata
        """
        # Retrieve larger pool for filtering
        candidate_k = top_k * 3
        
        # Get candidates from vector store
        raw_docs = self.vector_store.retrieve_top_k(query, k=candidate_k)
        
        # Convert to chunk format
        candidates = []
        for doc in raw_docs:
            candidates.append({
                'content': doc.page_content,
                'source': 'disease_knowledge.json',
                'source_type': 'JSON',
                'metadata': doc.metadata
            })
        
        # Apply metadata filters
        filtered = candidates
        
        if symptom_type_filter:
            # Match symptom type (partial match allowed)
            filtered = [
                c for c in filtered 
                if symptom_type_filter.lower() in c['metadata'].get('symptom_type', '').lower()
            ]
        
        if region_filter:
            # Match region (Karnataka or General)
            filtered = [
                c for c in filtered
                if c['metadata'].get('region', '').lower() == region_filter.lower() 
                or c['metadata'].get('region', '').lower() == 'general'
            ]
        
        # If filtering removed too many, fall back to unfiltered
        if len(filtered) < 2:
            filtered = candidates
        
        # Return top_k
        return filtered[:top_k]
    
    def retrieve_by_disease_name(self, disease_name: str) -> Optional[Dict]:
        """
        Retrieve specific disease by exact name
        
        Args:
            disease_name: Exact disease name
        
        Returns:
            Disease chunk or None
        """
        for chunk in self.chunks:
            if chunk['metadata']['disease_name'].lower() == disease_name.lower():
                return chunk
        return None
    
    def get_all_diseases(self) -> List[str]:
        """Get list of all disease names in database"""
        return [d['disease_name'] for d in self.diseases]
    
    def get_symptom_types(self) -> List[str]:
        """Get list of all symptom types"""
        types = set()
        for d in self.diseases:
            types.add(d['metadata']['symptom_type'])
        return sorted(list(types))
