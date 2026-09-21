"""
Diagnosis Generator Module
Generates final diagnosis and treatment recommendations
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from coffee_diagnosis.core.llm_client import LLMClient


@dataclass
class Diagnosis:
    """Diagnosis result"""
    disease_name: str
    confidence: float
    reason: str
    treatment: str
    prevention: str
    source: str


class DiagnosisGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize Diagnosis Generator

        Args:
            llm_client: Shared LLM client (defaults to local OpenAI-compatible endpoint)
        """
        self.llm = llm_client or LLMClient()

    def generate_diagnosis(
        self,
        query: str,
        user_responses: List[str],
        context: List[Dict],
        confidence: float
    ) -> Diagnosis:
        """
        Generate final diagnosis based on all collected information

        Args:
            query: Original user query
            user_responses: List of user's clarification answers
            context: Retrieved context chunks
            confidence: Current confidence score

        Returns:
            Diagnosis object with disease name, reasons, and treatment
        """
        # Format context and history
        context_text = self._format_context(context)
        history_text = self._format_history(query, user_responses)

        prompt = f"""You are an expert agricultural scientist specializing in coffee disease diagnosis in Karnataka region.

Based ONLY on the following information, provide a complete diagnosis with structured candidate comparison.

RETRIEVED KNOWLEDGE BASE:
{context_text}

USER INFORMATION:
{history_text}

DIAGNOSTIC RULES:
1. FIRST: List 2-3 candidate diseases with scores based on symptom match
2. THEN: Pick the best match and provide full diagnosis
3. Provide diagnosis ONLY based on the retrieved knowledge base above
4. Do NOT mention diseases not found in the knowledge base
5. Do NOT hallucinate symptoms or treatments
6. EXTRACT specific treatment and prevention DIRECTLY from knowledge base
7. No markdown headers (# ##) in treatment/prevention sections

FORMAT YOUR RESPONSE EXACTLY AS:

CANDIDATE COMPARISON:
1. [Disease 1]: [Score 0-100]% - [Evidence from symptoms]
2. [Disease 2]: [Score 0-100]% - [Evidence from symptoms]
3. [Disease 3]: [Score 0-100]% - [Evidence from symptoms]

FINAL DIAGNOSIS:
DISEASE: [name of highest scoring candidate]
CONFIDENCE: [0-100]%
SYMPTOMS MATCH: [list symptoms that match]
REASON: [why this diagnosis based on candidate comparison]
TREATMENT: [Extract specific treatment steps from knowledge base - NO markdown headers]
PREVENTION: [Extract specific prevention measures from knowledge base - NO markdown headers]
SOURCE PDF: [which PDF this information came from]

IMPORTANT: Finish every sentence properly. Extract FULL treatment/prevention from knowledge base.

Provide the structured diagnosis now:"""

        try:
            diagnosis_text = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )

            if not diagnosis_text:
                raise ValueError("Could not extract textual diagnosis from model response")

            diagnosis = self._parse_diagnosis(diagnosis_text, confidence, context)
            return diagnosis

        except Exception as e:
            print(f"[ERROR] Exception in generate_diagnosis: {str(e)}")
            import traceback
            traceback.print_exc()
            return Diagnosis(
                disease_name="Unable to diagnose",
                confidence=0.0,
                reason=f"Error during diagnosis: {str(e)}",
                treatment="Please consult a local agricultural expert",
                prevention="",
                source="Error"
            )

    def _format_context(self, context: List[Dict]) -> str:
        """Format context chunks"""
        formatted = []

        if not context:
            return "No context available"

        for i, chunk in enumerate(context[:5], 1):  # Use top 5 chunks
            try:
                # Handle both dict and Document objects
                if isinstance(chunk, dict):
                    content = chunk.get('content', '')
                    source = chunk.get('source', 'Unknown')
                else:
                    content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                    source = chunk.metadata.get('source_file', 'Unknown') if hasattr(chunk, 'metadata') else 'Unknown'

                formatted.append(f"[From {source}]\n{content}")
            except Exception as e:
                print(f"[ERROR] Error formatting context chunk {i}: {e}")
                continue

        return "\n\n---\n\n".join(formatted)

    def _format_history(self, query: str, responses: List[str]) -> str:
        """Format user information"""
        history = f"Initial Symptom Description:\n{query}\n\n"

        if responses:
            history += "Additional Clarifications:\n"
            for i, response in enumerate(responses, 1):
                history += f"{i}. {response}\n"

        return history

    def _parse_diagnosis(self, response_text: str, confidence: float, context: List[Dict] = None) -> Diagnosis:
        """Parse diagnosis from model response"""
        if not response_text:
            return Diagnosis(
                disease_name="Unable to determine",
                confidence=0.0,
                reason="No response from model",
                treatment="",
                prevention="",
                source=""
            )

        lines = response_text.split('\n')
        disease_name = "Unable to determine"
        reason = ""
        treatment = ""
        prevention = ""
        source = "Knowledge base"

        # Prefer explicitly reported DISEASE field
        for line in lines:
            if line.strip().lower().startswith("disease:"):
                disease_name = line.split(":", 1)[1].strip() or disease_name
                break

        # Coffee diseases database (expanded for hybrid JSON coverage)
        known_diseases = {
            'coffee leaf rust': ['rust', 'orange', 'powder', 'underside'],
            'brown eye spot': ['brown eye', 'eye spot', 'circular spots'],
            'anthracnose': ['anthracnose', 'sunken', 'berry', 'pink'],
            'root rot': ['root rot', 'brown roots', 'waterlogged', 'wilting'],
            'coffee berry borer': ['borer', 'berry', 'holes'],
            'scale insects': ['scale', 'honeydew', 'sooty'],
            'red spider mites': ['mite', 'mites', 'speckles', 'bronze'],
            'nitrogen deficiency': ['nitrogen', 'uniform yellowing', 'older leaves'],
            'magnesium deficiency': ['magnesium', 'interveinal', 'veins green'],
            'iron chlorosis': ['iron chlorosis', 'young leaves', 'veins remain green'],
            'potassium deficiency': ['potassium', 'leaf margins', 'crispy', 'necrosis'],
            'zinc deficiency': ['zinc', 'small leaves', 'rosette'],
            'phoma leaf spot': ['phoma', 'tip', 'margin', 'irregular brown'],
            'white stem borer': ['stem borer', 'sawdust', 'stem holes'],
            'coffee wilt disease': ['vascular', 'progressive wilting', 'top branches'],
        }

        # Convert text to lowercase for searching
        text_lower = response_text.lower()

        # Look for specific disease patterns in known_diseases
        best_match = None
        best_score = 0

        for disease, keywords in known_diseases.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > best_score:
                best_score = score
                best_match = disease

        if best_match and best_score > 0:
            disease_name = self._canonicalize_disease_name(best_match)
        else:
            # Fallback: look for disease keywords in order
            disease_patterns = [
                'coffee leaf rust',
                'leaf rust',
                'root rot',
                'anthracnose',
                'brown eye spot',
                'coffee berry borer',
                'berry borer',
                'scale insects',
                'scale insect',
                'red spider mites',
                'red spider mite',
                'nitrogen deficiency',
                'magnesium deficiency',
                'iron chlorosis',
                'potassium deficiency',
                'zinc deficiency',
                'phoma leaf spot',
                'white stem borer',
                'coffee wilt disease',
            ]

            for pattern in disease_patterns:
                if pattern in text_lower:
                    disease_name = self._canonicalize_disease_name(pattern)
                    break

        # Extract reason, treatment, prevention from specific sections
        full_text = response_text

        # Extract reason - find up to next section or new line pattern
        reason = "Based on symptoms provided"
        if 'reason:' in full_text.lower():
            reason_start = full_text.lower().find('reason:') + len('reason:')
            reason_end = full_text.lower().find('\n-', reason_start)
            if reason_end == -1:
                reason_end = full_text.lower().find('symptoms match:', reason_start)
            if reason_end == -1:
                reason_end = len(full_text)
            reason = full_text[reason_start:reason_end].strip()[:500]

        # Extract treatment - FULL extraction from knowledge base
        treatment = ""
        if 'treatment:' in full_text.lower():
            treatment_start = full_text.lower().find('treatment:') + len('treatment:')
            treatment_end = full_text.lower().find('\n-', treatment_start)
            if treatment_end == -1:
                treatment_end = full_text.lower().find('prevention:', treatment_start)
            if treatment_end == -1:
                treatment_end = full_text.lower().find('source pdf:', treatment_start)
            if treatment_end == -1:
                treatment_end = len(full_text)
            treatment = full_text[treatment_start:treatment_end].strip()[:1000]
            # Remove markdown headers that cause formatting issues
            treatment = self._clean_markdown_headers(treatment)

        # If treatment is too short or empty, it means knowledge base wasn't extracted
        if not treatment or len(treatment.strip()) < 20:
            treatment = "Treatment information available in knowledge base - Please consult local agricultural extension office for implementation"

        # Extract prevention - FULL extraction from knowledge base
        prevention = ""
        if 'prevention:' in full_text.lower():
            prevention_start = full_text.lower().find('prevention:') + len('prevention:')
            prevention_end = full_text.lower().find('\n-', prevention_start)
            if prevention_end == -1:
                prevention_end = full_text.lower().find('source pdf:', prevention_start)
            if prevention_end == -1:
                prevention_end = len(full_text)
            prevention = full_text[prevention_start:prevention_end].strip()[:1000]
            # Remove markdown headers that cause formatting issues
            prevention = self._clean_markdown_headers(prevention)

        # If prevention is too short or empty, it means knowledge base wasn't extracted
        if not prevention or len(prevention.strip()) < 20:
            prevention = "Prevention measures available in knowledge base - Implement preventive practices based on retrieved information"

        return Diagnosis(
            disease_name=disease_name,
            confidence=confidence,
            reason=reason if reason else "Based on symptoms provided",
            treatment=treatment if treatment else "Consult agricultural expert",
            prevention=prevention if prevention else "Preventive measures recommended",
            source=source
        )

    def _canonicalize_disease_name(self, name: str) -> str:
        """Map aliases to canonical disease names."""
        n = (name or "").strip().lower()
        alias_map = {
            "leaf rust": "Coffee Leaf Rust",
            "coffee leaf rust": "Coffee Leaf Rust",
            "berry borer": "Coffee Berry Borer",
            "coffee berry borer": "Coffee Berry Borer",
            "scale insect": "Scale Insects",
            "scale insects": "Scale Insects",
            "red spider mite": "Red Spider Mites",
            "red spider mites": "Red Spider Mites",
        }
        if n in alias_map:
            return alias_map[n]
        return (name or "").strip().title()

    def _clean_markdown_headers(self, text: str) -> str:
        """Remove markdown headers (# ## ###) that cause large font rendering in UI"""
        import re
        # Remove markdown headers at the start of lines
        # Matches: # Header, ## Header, ### Header, etc.
        cleaned = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        return cleaned.strip()

    def format_diagnosis_output(self, diagnosis: Diagnosis) -> str:
        """Format diagnosis for display"""
        output = f"""
===================================================
                DIAGNOSIS RESULT
===================================================

Disease: {diagnosis.disease_name}
Confidence: {diagnosis.confidence * 100:.1f}%

Reason:
{diagnosis.reason}

Treatment:
{diagnosis.treatment}

Prevention:
{diagnosis.prevention}

Source: {diagnosis.source}
===================================================
"""
        return output
