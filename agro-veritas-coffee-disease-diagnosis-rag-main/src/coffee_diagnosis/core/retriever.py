"""
Hybrid Retriever Module with JSON + PDF Sources
Combines structured JSON disease knowledge (60%) with PDF evidence (40%)
Multi-query expansion, hybrid scoring, and source diversity
"""

from typing import List, Dict, Optional, Tuple
from .vector_store import FAISSVectorStore
from .json_retriever import JSONRetriever
import re
from collections import defaultdict


class Retriever:
    def __init__(
        self, 
        vector_store: FAISSVectorStore, 
        json_retriever: Optional[JSONRetriever] = None,
        top_k: int = 5
    ):
        """
        Initialize Hybrid Retriever (JSON + PDF)

        Args:
            vector_store: FAISSVectorStore instance for PDF documents
            json_retriever: JSONRetriever instance for structured disease knowledge
            top_k: Number of final documents to return
        """
        self.vector_store = vector_store  # PDF retriever
        self.json_retriever = json_retriever  # JSON retriever
        self.top_k = top_k
        self.candidate_pool_multiplier = 4  # Retrieve 4x candidates before filtering
        
        # Hybrid ratios
        self.json_ratio = 0.6  # 60% from JSON
        self.pdf_ratio = 0.4   # 40% from PDF

    def retrieve(
        self, 
        query: str, 
        previous_answers: List[str] = None,
        symptom_type_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Hybrid retrieval: JSON (60%) + PDF (40%) with multi-query expansion

        Args:
            query: Current user query
            previous_answers: List of previous user answers (conversation history)
            symptom_type_filter: Optional filter by symptom type for JSON retrieval

        Returns:
            List of top_k documents with hybrid scores, deduplicated and diversified
        """
        # Calculate split: how many from each source
        json_count = int(self.top_k * self.json_ratio * self.candidate_pool_multiplier)
        pdf_count = int(self.top_k * self.pdf_ratio * self.candidate_pool_multiplier)
        
        all_candidates = []
        
        # 1. Retrieve from JSON (if available)
        if self.json_retriever:
            json_chunks = self.json_retriever.retrieve(
                query=query,
                top_k=json_count,
                symptom_type_filter=symptom_type_filter
            )
            # Tag as JSON source
            for chunk in json_chunks:
                chunk['source_type'] = 'JSON'
            all_candidates.extend(json_chunks)
        
        # 2. Retrieve from PDFs with multi-query expansion
        query_variants = self._generate_query_variants(query, previous_answers or [])
        
        for variant in query_variants:
            docs = self.vector_store.retrieve_top_k(variant, k=pdf_count)
            for doc in docs:
                all_candidates.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source_file', 'unknown'),
                    'source_type': 'PDF',
                    'metadata': doc.metadata,
                    'query_variant': variant
                })
        
        # 3. Calculate hybrid scores
        scored_chunks = self._calculate_hybrid_scores(all_candidates, query_variants)
        
        # Deduplicate by content
        unique_chunks = self._deduplicate_chunks(scored_chunks)
        
        # Enforce source diversity
        diverse_chunks = self._enforce_diversity(unique_chunks, max_per_source=2)
        
        # Sort by hybrid score and return top_k
        diverse_chunks.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        return diverse_chunks[:self.top_k]

    def retrieve_with_scores(self, query: str, previous_answers: List[str] = None) -> List[Dict]:
        """
        Wrapper for compatibility - same as retrieve()
        """
        return self.retrieve(query, previous_answers)

    def _generate_query_variants(self, query: str, previous_answers: List[str]) -> List[str]:
        """
        Generate multiple query variants for multi-query retrieval

        Returns:
            List of query variants: [base_query, base+history, keyword_query]
        """
        variants = []
        
        # Variant 1: Base query
        variants.append(query)
        
        # Variant 2: Base + conversation history
        if previous_answers:
            context_str = " ".join(previous_answers)
            variants.append(f"{query} {context_str}")
        
        # Variant 3: Condensed keywords from query + history
        keywords = self._extract_keywords(query, previous_answers)
        if keywords:
            variants.append(" ".join(keywords))
        
        return variants

    def _extract_keywords(self, query: str, previous_answers: List[str]) -> List[str]:
        """
        Extract key symptom/disease-related keywords

        Returns:
            List of important keywords
        """
        combined_text = query + " " + " ".join(previous_answers or [])
        
        # Key symptom/disease terms
        keywords = []
        symptom_terms = [
            'yellow', 'brown', 'black', 'white', 'orange', 'red', 'green',
            'spot', 'spots', 'leaf', 'leaves', 'stem', 'root', 'berry', 'fruit',
            'wilting', 'dying', 'falling', 'curling', 'drooping',
            'powder', 'mold', 'fungus', 'rot', 'blight', 'rust',
            'ring', 'pattern', 'margin', 'edge', 'vein', 'tip'
        ]
        
        words = combined_text.lower().split()
        for term in symptom_terms:
            if term in words:
                keywords.append(term)
        
        return keywords[:10]  # Limit to top 10

    def _calculate_hybrid_scores(self, chunks: List[Dict], query_variants: List[str]) -> List[Dict]:
        """
        Calculate hybrid scores combining vector similarity, lexical overlap, and variant coverage

        Scoring:
        - Vector similarity: 55%
        - Lexical overlap: 35%
        - Multi-variant hit: 10%
        """
        for chunk in chunks:
            # Vector score (normalized 0-1, already from FAISS)
            vector_score = 0.5  # Placeholder (FAISS doesn't return scores by default)
            
            # Lexical overlap score
            lexical_score = self._calculate_lexical_overlap(chunk['content'], query_variants)
            
            # Variant hit bonus (appears in multiple query variants)
            variant_score = 0.1 if len(query_variants) > 1 else 0
            
            # Hybrid score
            chunk['hybrid_score'] = (
                vector_score * 0.55 +
                lexical_score * 0.35 +
                variant_score * 0.10
            )
        
        return chunks

    def _calculate_lexical_overlap(self, content: str, query_variants: List[str]) -> float:
        """
        Calculate lexical overlap between content and queries

        Returns:
            Overlap score (0-1)
        """
        content_lower = content.lower()
        all_query_words = set()
        
        for variant in query_variants:
            words = re.findall(r'\b\w+\b', variant.lower())
            all_query_words.update(w for w in words if len(w) > 3)  # Ignore short words
        
        if not all_query_words:
            return 0.0
        
        matches = sum(1 for word in all_query_words if word in content_lower)
        return min(matches / len(all_query_words), 1.0)

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate chunks by content prefix

        Uses first 220 characters (normalized) as dedup key
        """
        seen_prefixes = set()
        unique_chunks = []
        
        for chunk in chunks:
            content = chunk['content']
            # Normalize: lowercase, remove extra spaces
            normalized = re.sub(r'\s+', ' ', content.lower()).strip()
            prefix = normalized[:220]
            
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                unique_chunks.append(chunk)
        
        return unique_chunks

    def _enforce_diversity(self, chunks: List[Dict], max_per_source: int = 2) -> List[Dict]:
        """
        Enforce source diversity - limit chunks per source file

        Args:
            chunks: List of chunks
            max_per_source: Maximum chunks allowed from same source

        Returns:
            Filtered chunks with diversity enforced
        """
        source_counts = defaultdict(int)
        diverse_chunks = []
        
        for chunk in chunks:
            source = chunk.get('source', 'unknown')
            if source_counts[source] < max_per_source:
                diverse_chunks.append(chunk)
                source_counts[source] += 1
        
        return diverse_chunks

        return results

    def retrieve_with_scores(self, query: str, previous_answers: List[str] = None) -> List[Dict]:
        """
        Retrieve documents with relevance scores

        Args:
            query: Current user query
            previous_answers: List of previous user answers

        Returns:
            List of retrieved documents with relevance scores
        """
        if previous_answers:
            context_str = " ".join(previous_answers)
            combined_query = f"{query} {context_str}"
        else:
            combined_query = query

        # Search with scores
        search_results = self.vector_store.search(combined_query, k=self.top_k)

        results = []
        for doc, score in search_results:
            results.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source_file', 'unknown'),
                'metadata': doc.metadata,
                'score': score  # Lower score = more similar
            })

        return results

    def format_context(self, documents: List[Dict], max_length: int = 2000) -> str:
        """
        Format retrieved documents into context string

        Args:
            documents: List of retrieved documents
            max_length: Maximum total length

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for doc in documents:
            content = doc['content']
            source = doc['source']

            text = f"Source: {source}\n{content}\n"

            if current_length + len(text) > max_length:
                break

            context_parts.append(text)
            current_length += len(text)

        return "\n---\n".join(context_parts)
