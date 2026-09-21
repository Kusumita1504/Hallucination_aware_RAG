"""
Clarification Generator Module (RAC)
Generates clarification questions grounded in retrieved context
"""

from typing import List, Dict, Optional, Set
import re
import json
from pathlib import Path
from coffee_diagnosis.core.llm_client import LLMClient


class ClarificationGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None, disease_priorities_path: Optional[str] = None):
        """
        Initialize Clarification Generator

        Args:
            llm_client: Shared LLM client (defaults to local OpenAI-compatible endpoint)
            disease_priorities_path: Path to disease_question_priorities.json (defaults to data/ dir)
        """
        self.llm = llm_client or LLMClient()
        self.disease_priorities = self._load_disease_priorities(disease_priorities_path)
        self.asked_priority_questions: Set[str] = set()  # Track which priority questions have been asked

    def _load_disease_priorities(self, custom_path: Optional[str] = None) -> Dict:
        """
        Load disease-specific question priorities from JSON file

        Args:
            custom_path: Custom path to disease_question_priorities.json (for testing)

        Returns:
            Dictionary of disease priorities
        """
        if custom_path:
            path = Path(custom_path)
        else:
            # Try multiple locations
            possible_paths = [
                Path("data/disease_question_priorities.json"),
                Path("../data/disease_question_priorities.json"),
                Path("../../data/disease_question_priorities.json"),
            ]
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    break

        if path and path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    return data.get('priority_questions', {})
            except Exception as e:
                print(f"[WARN] Failed to load disease priorities from {path}: {e}")
                return {}
        else:
            print("[WARN] disease_question_priorities.json not found, using generic questions only")
            return {}

    def _get_disease_priority_question(
        self,
        suspected_diseases: List[str],
        missing_keys: List[str],
        user_text: str
    ) -> Optional[str]:
        """
        Get the next priority question for one of the suspected diseases

        Args:
            suspected_diseases: List of likely disease names
            missing_keys: List of missing info keys
            user_text: User's symptom description so far

        Returns:
            A priority question for the suspected disease, or None if no match
        """
        if not self.disease_priorities or not suspected_diseases:
            return None

        # Try to find a priority question from the top suspected disease
        for disease in suspected_diseases:
            if disease not in self.disease_priorities:
                continue

            priorities = self.disease_priorities[disease]
            # Try each priority question
            for priority_level in ['priority_1', 'priority_2', 'priority_3', 'priority_4', 'priority_5']:
                if priority_level in priorities:
                    question = priorities[priority_level]

                    # Skip if already asked
                    if question in self.asked_priority_questions:
                        continue

                    # Verify the question isn't assuming symptoms not mentioned
                    if self._is_assumptive(question, user_text):
                        continue

                    # Mark as asked and return
                    self.asked_priority_questions.add(question)
                    return question

        return None

    def reset_priority_tracking(self):
        """Reset tracked priority questions (useful for new diagnosis session)"""
        self.asked_priority_questions = set()



    def generate_question(
        self,
        query: str,
        context: List[Dict],
        previous_answers: List[str],
        missing_info: Dict,
        max_questions: int = 3,
        suspected_diseases: Optional[List[str]] = None
    ) -> str:
        """
        Generate a clarification question based on retrieved context and what's missing

        Args:
            query: Original user query
            context: Retrieved context chunks
            previous_answers: List of previous user answers
            missing_info: Missing symptom information (keys only, like 'color', 'pattern')
            max_questions: Maximum questions already asked
            suspected_diseases: List of likely diseases (for priority-based questions)

        Returns:
            Generated clarification question
        """
        # Format context
        context_text = self._format_context(context)
        history_text = self._format_history(query, previous_answers)

        # Create a concise list of what's missing
        missing_keys = list(missing_info.keys()) if missing_info else []
        user_text = f"{query} {' '.join(previous_answers)}".lower()

        # Off-topic guard: if the user hasn't mentioned coffee plants or plant symptoms, redirect
        combined_text = f"{query} {' '.join(previous_answers)}".lower()
        plant_terms = ["coffee", "leaf", "leaves", "plant", "tree", "berry", "berries", "stem"]
        if not any(term in combined_text for term in plant_terms):
            return (
                "I can help with coffee plant health. Please describe the coffee plant symptoms "
                "(e.g., leaf color/pattern, spots, wilting, affected parts like leaves, stems, or berries)."
            )

        # TRY DISEASE-SPECIFIC PRIORITY QUESTIONS FIRST
        if suspected_diseases and missing_keys:
            priority_question = self._get_disease_priority_question(
                suspected_diseases, missing_keys, user_text
            )
            if priority_question:
                return priority_question

        if not missing_keys:
            return "Do you see any other symptoms or changes on the plant?"

        # Use Claude to generate smart follow-up questions
        prompt = f"""You are helping a farmer diagnose their coffee plant problem. Ask ONE clear, specific question.

User's symptoms so far:
{history_text}

What's missing: {', '.join(missing_keys)}

RULES FOR YOUR QUESTION:
1. Maximum 20-25 words - be clear and specific
2. Ask about ONE thing only (not multiple things)
3. Use simple, practical language - avoid overly technical terms
4. End with exactly ONE question mark (?)
5. NEVER use ?? or multiple question marks
6. Ask for observable details the farmer can see
7. Use ONLY symptoms already mentioned by the user. Do NOT introduce new symptoms.
8. If user only said "wilting", ask about wilting details only (not yellowing/spots unless user said them).
9. Examples of GOOD questions:
   - "Are the yellow leaves appearing on the older lower branches or on the newer growth at the top?"
   - "Do the affected leaves have distinct spots with defined edges, or is the discoloration more spread out?"
   - "Did this yellowing start suddenly within the last few days, or has it been developing slowly over weeks?"
   - "Is the wilting affecting the whole plant or only some branches?"

10. BAD - DO NOT copy these patterns:
   - Using ?? at the end (NEVER DO THIS)
   - Overly technical disease names or scientific terminology
   - Multiple sub-questions bundled together
   - Asking about yellowing/spots when user only reported wilting

Generate ONE clear question (20-25 words, ending with one ?):"""

        try:
            question = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
            )

            question = self._clean_question(question)
            # Guard against introducing symptoms user never mentioned
            if self._is_assumptive(question, user_text) or self._is_low_quality_question(question):
                return self._generate_fallback_question(missing_keys, user_text=user_text)
            return question

        except Exception as e:
            print(f"[ERROR] Failed to generate question: {str(e)}")
            # Fallback to a generic but smart question
            return self._generate_fallback_question(missing_keys, user_text=user_text)

    def _format_context(self, context: List[Dict]) -> str:
        """Format context chunks for the prompt"""
        formatted = []

        for i, chunk in enumerate(context[:3], 1):  # Use top 3
            source = chunk.get('source', 'Unknown')
            content = chunk['content'][:300]  # Truncate to 300 chars

            formatted.append(f"[{i}] Source: {source}\n{content}...")

        return "\n\n".join(formatted)

    def _format_history(self, query: str, previous_answers: List[str]) -> str:
        """Format conversation history"""
        if not previous_answers:
            return f"Initial observation: {query}"

        history = f"Initial symptom: {query}\n"

        for i, answer in enumerate(previous_answers, 1):
            history += f"Follow-up {i}: {answer}\n"

        return history

    def _generate_fallback_question(self, missing_info, user_text: str = "") -> str:
        """Generate a smart fallback question when API fails"""
        # missing_info is now a list of keys like ['color', 'pattern']
        if isinstance(missing_info, dict):
            missing_keys = list(missing_info.keys())
        else:
            missing_keys = missing_info if isinstance(missing_info, list) else []

        # Simple, short questions (max 15 words each)
        fallback_questions = {
            'color': "What color is the problem area - yellow, brown, black, or red?",
            'pattern': "Are there distinct spots or is the discoloration spread out?",
            'location': "Which leaves are affected - older ones or newer growth?",
            'spread': "Where on the plant do you see the symptoms?",
            'timing': "Did this appear suddenly or develop slowly over time?",
            'texture': "Do affected areas look dry, wet, or powdery?",
            'berry_fruit': "Are the coffee berries showing any problems?",
        }

        # If user mentioned wilting only, keep follow-up strictly about wilting details
        if "wilt" in (user_text or "") and "yellow" not in (user_text or "") and "spot" not in (user_text or ""):
            wilt_specific = {
                'location': "Is the wilting affecting the whole plant or only some branches?",
                'spread': "Did the wilting spread gradually or suddenly across the plant?",
                'timing': "When did the wilting first start?",
                'pattern': "Is the wilting constant all day or worse at certain times?"
            }
            if missing_keys and missing_keys[0] in wilt_specific:
                return wilt_specific[missing_keys[0]]

        # Ask about the first missing attribute
        if missing_keys:
            return fallback_questions.get(missing_keys[0], "Can you describe what you see in more detail?")

        return "Do you notice any other unusual changes on the plant?"

    def _is_assumptive(self, question: str, user_text: str) -> bool:
        """Detect if generated question introduces symptoms absent in user description."""
        if not question:
            return False
        q = question.lower()
        user = user_text.lower()
        symptom_terms = ["yellow", "yellowing", "spot", "spots", "powder", "orange", "ring", "margin", "vein"]
        for term in symptom_terms:
            if term in q and term not in user:
                return True
        return False

    def _clean_question(self, question: str) -> str:
        """Keep only the main question line and ensure it ends with a question mark."""
        if not question:
            return "Can you share more details about the symptoms?"

        # Remove prefixed labels and reasoning/explanations after newlines
        if question.lower().startswith("question:"):
            question = question.split(":", 1)[1].strip()

        # Take the first non-empty line as the question
        lines = [l.strip() for l in question.splitlines() if l.strip()]
        question = lines[0] if lines else question.strip()

        # Normalize whitespace and remove obvious control/special chars
        question = re.sub(r'\s+', ' ', question).strip()
        question = re.sub(r'[^\w\s,\-\'"?/()]', '', question)

        # Trim trailing reasoning markers
        for marker in ["reasoning:", "explanation:", "because"]:
            idx = question.lower().find(marker)
            if idx > 0:
                question = question[:idx].strip()

        # Fix malformed endings
        # First, remove any trailing quotes (but preserve question marks inside them)
        # Then ensure proper question mark at the end
        
        # Remove double/triple question marks: text?? or text??? → text?
        while question.endswith('??'):
            question = question[:-1]

        # Collapse repeated question marks anywhere in the sentence
        question = re.sub(r'\?{2,}', '?', question)
        
        # Remove patterns like: text"? → text?
        if question.endswith('"?'):
            question = question[:-2] + '?'
        elif question.endswith("'?"):
            question = question[:-2] + '?'
        # Remove patterns like: text." or text.' → text?
        elif question.endswith('."'):
            question = question[:-2] + '?'
        elif question.endswith(".'"):
            question = question[:-2] + '?'
        # Remove trailing quotes without question mark: text" → text?
        elif question.endswith('"'):
            question = question[:-1] + '?'
        elif question.endswith("'"):
            question = question[:-1] + '?'
        # Ensure question mark at end
        elif not question.endswith("?"):
            question = question.rstrip(".") + "?"

        # Keep only first question if model produced multiple sentences/questions
        if question.count("?") > 1:
            question = question.split("?", 1)[0].strip() + "?"

        # Keep question concise for UI readability
        words = question.split()
        if len(words) > 28:
            question = " ".join(words[:28]).rstrip(",.") + "?"

        return question

    def _is_low_quality_question(self, question: str) -> bool:
        """
        Detect malformed/gibberish questions and force deterministic fallback.
        """
        if not question:
            return True

        q = question.strip()
        if len(q) < 8:
            return True

        # Must end as a single question
        if not q.endswith("?") or q.count("?") != 1:
            return True

        words = re.findall(r"[A-Za-z']+", q)
        if len(words) < 4:
            return True

        # Detect strange tokens like YOUTUIONING / random long uppercase words
        for w in words:
            wl = w.lower()
            if len(w) >= 9 and w.isupper():
                return True
            if len(w) >= 10 and len(re.findall(r"[aeiou]", wl)) <= 1:
                return True

        return False

    def validate_question(self, question: str, context: List[Dict]) -> bool:
        """
        Validate that question is grounded in context

        Args:
            question: Question to validate
            context: Retrieved context

        Returns:
            True if question appears grounded, False otherwise
        """
        question_lower = question.lower()
        context_text = " ".join([c['content'].lower() for c in context])

        # Check if key words from question appear in context
        words = [w for w in question_lower.split() if len(w) > 3]
        matches = sum(1 for word in words if word in context_text)

        return matches > 0
