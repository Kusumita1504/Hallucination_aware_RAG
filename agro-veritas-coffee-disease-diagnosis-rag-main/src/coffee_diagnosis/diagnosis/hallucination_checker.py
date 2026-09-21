"""
Hallucination Checker Module (SelfCheckGPT)
Verifies consistency of diagnoses to detect hallucinations
"""

from typing import List, Dict, Optional
from coffee_diagnosis.core.llm_client import LLMClient
from collections import Counter


class HallucinationChecker:
    def __init__(self, llm_client: Optional[LLMClient] = None, num_generations: int = 1):
        """
        Initialize Hallucination Checker

        Args:
            llm_client: Shared LLM client (defaults to local OpenAI-compatible endpoint)
            num_generations: Number of independent diagnosis generations (kept low for rate limits)
        """
        self.llm = llm_client or LLMClient()
        self.num_generations = num_generations

    def check_hallucination(
        self,
        query: str,
        user_responses: List[str],
        context: List[Dict],
        initial_diagnosis: str
    ) -> Dict:
        """
        Check for hallucinations by generating diagnosis multiple times

        Args:
            query: Original user query
            user_responses: List of user responses
            context: Retrieved context
            initial_diagnosis: Initial diagnosis result

        Returns:
            Dictionary with consistency scores and warnings
        """
        print(f"Generating {self.num_generations} independent diagnoses for verification...")

        diagnoses = []

        for i in range(self.num_generations):
            diagnosis = self._generate_diagnosis(query, user_responses, context)
            diagnoses.append(diagnosis)
            print(f"  Generation {i+1} complete")

        # Analyze consistency
        consistency_report = self._analyze_consistency(diagnoses, initial_diagnosis)

        return consistency_report

    def _generate_diagnosis(
        self,
        query: str,
        user_responses: List[str],
        context: List[Dict]
    ) -> str:
        """Generate a single diagnosis"""
        context_text = self._format_context(context)
        history_text = self._format_history(query, user_responses)

        prompt = f"""You are a coffee disease expert. Based ONLY on the information provided, diagnose the disease.

CONTEXT:
{context_text}

USER INFORMATION:
{history_text}

RULES:
1. Use ONLY information from the context
2. Do NOT hallucinate diseases or treatments
3. Be concise
4. State the disease name clearly

Diagnosis (just the disease name and key symptoms)."""

        try:
            diagnosis_text = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            if not diagnosis_text:
                diagnosis_text = "No diagnosis generated"
            return diagnosis_text

        except Exception as e:
            return f"Error: {str(e)}"

    def _format_context(self, context: List[Dict]) -> str:
        """Format context"""
        formatted = []
        for chunk in context[:3]:
            try:
                if isinstance(chunk, dict):
                    content = chunk.get('content', '')[:200]
                else:
                    content = str(chunk)[:200]
                formatted.append(content)
            except Exception as e:
                formatted.append(f"[Error processing chunk: {str(e)}]")
        return "\n\n".join(formatted)

    def _format_history(self, query: str, responses: List[str]) -> str:
        """Format history"""
        history = query
        for response in responses:
            history += f"\nUser: {response}"
        return history

    def _analyze_consistency(
        self,
        diagnoses: List[str],
        initial_diagnosis: str
    ) -> Dict:
        """
        Analyze consistency across multiple diagnoses

        Args:
            diagnoses: List of generated diagnoses
            initial_diagnosis: Initial diagnosis

        Returns:
            Consistency report
        """
        # Extract and normalize disease names from each generation
        disease_names = []
        for diagnosis in diagnoses:
            if diagnosis and diagnosis.strip():
                extracted = self._extract_disease_name(diagnosis)
                disease_names.append(self._normalize_disease_name(extracted))

        if not disease_names:
            return {
                'consistent': False,
                'consistency_score': 0.0,
                'disease_name': "Unknown",
                'all_diseases': [],
                'hallucination_detected': True,
                'warnings': ["[WARNING] Could not extract diagnosis from verification generations"],
                'num_generations': self.num_generations,
                'agreement': "0%",
                'safe_to_finalize': False
            }

        counts = Counter(disease_names)
        majority_disease, majority_count = counts.most_common(1)[0]
        consistency_score = majority_count / max(len(disease_names), 1)
        consistent_diseases = set(disease_names)

        warnings = []
        # Flag as hallucination only when no strong majority agreement.
        hallucination_detected = consistency_score < 0.67
        if hallucination_detected:
            warnings.append(
                f"[WARNING] Low diagnosis agreement across generations: {dict(counts)}"
            )

        # Safety signal for active gating
        safe_to_finalize = consistency_score >= 0.67 and not hallucination_detected

        return {
            'consistent': not hallucination_detected,
            'consistency_score': consistency_score,
            'disease_name': majority_disease if majority_disease else "Unknown",
            'all_diseases': list(consistent_diseases),
            'hallucination_detected': hallucination_detected,
            'warnings': warnings,
            'num_generations': self.num_generations,
            'agreement': f"{int(consistency_score * 100)}%",
            'safe_to_finalize': safe_to_finalize
        }

    def _extract_disease_name(self, diagnosis_text: str) -> str:
        """Extract disease name from diagnosis text"""
        text = diagnosis_text.lower()

        known_patterns = [
            "coffee leaf rust",
            "leaf rust",
            "coffee berry borer",
            "brown eye spot",
            "anthracnose",
            "root rot",
            "coffee wilt disease",
            "white stem borer",
            "red spider mites",
            "red spider mite",
            "scale insects",
            "scale insect",
            "nitrogen deficiency",
            "magnesium deficiency",
            "iron chlorosis",
            "zinc deficiency",
            "potassium deficiency",
            "phoma leaf spot",
        ]

        for pattern in known_patterns:
            if pattern in text:
                return pattern.title()

        # Fallback to first line if no known disease phrase is found
        lines = diagnosis_text.split('\n')
        if lines:
            return lines[0].strip()
        return "Unknown"

    def _normalize_disease_name(self, disease_name: str) -> str:
        """Normalize aliases to canonical disease names."""
        if not disease_name:
            return "Unknown"
        d = disease_name.strip().lower()
        alias_map = {
            "leaf rust": "Coffee Leaf Rust",
            "coffee leaf rust": "Coffee Leaf Rust",
            "berry borer": "Coffee Berry Borer",
            "coffee berry borer": "Coffee Berry Borer",
            "red spider mite": "Red Spider Mites",
            "red spider mites": "Red Spider Mites",
            "scale insect": "Scale Insects",
            "scale insects": "Scale Insects",
        }
        if d in alias_map:
            return alias_map[d]
        return disease_name.strip().title()

    def _check_treatment_consistency(self, diagnoses: List[str]) -> float:
        """Check if treatment recommendations are similar"""
        # Simple check: count how many diagnoses mention treatment-related keywords
        treatment_keywords = ['treatment', 'spray', 'fungicide', 'manage', 'control', 'apply']
        mention_count = 0

        for diagnosis in diagnoses:
            diagnosis_lower = diagnosis.lower()
            if any(keyword in diagnosis_lower for keyword in treatment_keywords):
                mention_count += 1

        return mention_count / len(diagnoses) if diagnoses else 0.0

    def _extract_symptoms(self, diagnoses: List[str]) -> List[str]:
        """Extract mentioned symptoms across all diagnoses"""
        symptom_keywords = [
            'yellow', 'brown', 'spot', 'leaf', 'stem', 'rust', 'borer', 'scale',
            'rot', 'wilt', 'pattern', 'disease'
        ]

        symptoms = set()
        for diagnosis in diagnoses:
            diagnosis_lower = diagnosis.lower()
            for symptom in symptom_keywords:
                if symptom in diagnosis_lower:
                    symptoms.add(symptom)

        return list(symptoms)

    def format_verification_report(self, report: Dict) -> str:
        """Format verification report for display"""
        status = "[OK] VERIFIED" if report['consistent'] else "[WARNING] ISSUE DETECTED"

        output = f"""
{"="*50}
{status} - Hallucination Check
{"="*50}

Consistency Score: {report['consistency_score'] * 100:.1f}%
Agreement Level: {report['agreement']}

Disease Agreement: {', '.join(report['all_diseases'])}

{f"Warnings:" + chr(10) + chr(10).join(report['warnings']) if report['warnings'] else "No issues detected"}

Generations Checked: {report['num_generations']}
{"="*50}
"""
        return output
