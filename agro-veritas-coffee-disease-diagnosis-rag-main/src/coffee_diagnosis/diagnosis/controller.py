"""
Multi-turn Controller Module
Orchestrates the complete multi-turn coffee disease diagnosis pipeline
"""

from typing import Optional, Dict, List
from dotenv import load_dotenv
from coffee_diagnosis.core.pdf_loader import PDFLoader
from coffee_diagnosis.core.vector_store import FAISSVectorStore
from coffee_diagnosis.core.retriever import Retriever
from coffee_diagnosis.core.json_retriever import JSONRetriever
from coffee_diagnosis.rag.ambiguity_detector import AmbiguityDetector
from coffee_diagnosis.rag.retrieval_evaluator import RetrievalEvaluator
from coffee_diagnosis.rag.clarification_gen import ClarificationGenerator
from coffee_diagnosis.rag.state_manager import StateManager
from .diagnosis_generator import DiagnosisGenerator, Diagnosis
from .hallucination_checker import HallucinationChecker
from coffee_diagnosis.core.llm_client import LLMClient
from config import settings
import os
from pathlib import Path

# Load environment variables
load_dotenv()


class CoffeeDiagnosisController:
    def __init__(self, data_dir: str = ".", vector_db_path: str = "vector_db"):
        """
        Initialize the complete diagnosis system

        Args:
            data_dir: Directory containing PDF files
            vector_db_path: Path for FAISS vector store
        """
        print("[INIT] Initializing Coffee Disease Diagnosis System...")

        # Initialize LLM client (defaults to local OpenAI-compatible endpoint)
        llm_client = LLMClient()

        # Initialize components with API key (Claude Haiku by default for lower rate limits)
        self.ambiguity_detector = AmbiguityDetector()
        self.retrieval_evaluator = RetrievalEvaluator(relevance_threshold=0.3)
        self.clarification_gen = ClarificationGenerator(llm_client=llm_client)
        self.state_manager = StateManager()
        self.diagnosis_gen = DiagnosisGenerator(llm_client=llm_client)
        self.hallucination_checker = HallucinationChecker(
            llm_client=llm_client,
            num_generations=settings.NUM_GENERATIONS_FOR_VERIFICATION
        )

        # Initialize vector store for PDFs (load cache if unchanged, else rebuild)
        print("[LOAD] Initializing PDF vector store...")
        self.vector_store = FAISSVectorStore(vector_db_path=vector_db_path)

        pdf_signature = self._compute_pdf_signature(data_dir)
        if not self.vector_store.load_index(expected_signature=pdf_signature):
            print("[LOAD] Cache miss/stale. Loading PDFs and creating vector store...")
            pdf_loader = PDFLoader(data_dir=data_dir)
            documents = pdf_loader.load_pdfs()
            self.vector_store.create_index(documents, source_signature=pdf_signature)
        else:
            print("[LOAD] Reusing cached PDF index")

        # Initialize JSON retriever for structured disease knowledge
        # Support both data_dir=".../data" and data_dir=".../data/pdfs"
        if os.path.basename(os.path.normpath(data_dir)).lower() == "pdfs":
            json_base_dir = os.path.dirname(data_dir)
        else:
            json_base_dir = data_dir
        json_path = os.path.join(json_base_dir, 'disease_knowledge.json')
        json_retriever = None
        if os.path.exists(json_path):
            print("[LOAD] Loading structured JSON disease knowledge...")
            json_retriever = JSONRetriever(
                json_path=json_path,
                embeddings=self.vector_store.embedding_model
            )
            print(f"[OK] Loaded {len(json_retriever.diseases)} diseases from JSON")
        else:
            print(f"[WARN] JSON knowledge base not found at {json_path}, using PDF only")

        # Initialize hybrid retriever (JSON + PDF)
        self.retriever = Retriever(
            vector_store=self.vector_store,
            json_retriever=json_retriever,
            top_k=5
        )

        print("[OK] System initialized successfully!\n")

    def _compute_pdf_signature(self, data_dir: str) -> str:
        """Build a simple freshness signature from PDF names + modified times + sizes."""
        pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
        parts = []
        for p in pdf_paths:
            stat = p.stat()
            parts.append(f"{p.name}:{int(stat.st_mtime)}:{stat.st_size}")
        return "|".join(parts)

    def _predict_likely_diseases(self, query: str, context: List[Dict]) -> List[str]:
        """
        Predict the most likely diseases based on initial query and retrieved context

        Args:
            query: User's initial symptom query
            context: Retrieved context chunks

        Returns:
            List of likely disease names, ordered by probability
        """
        # Extract disease mentions from context
        diseases_mentioned = set()
        query_lower = query.lower()

        # Check retrieved documents for disease names
        for doc in context:
            content = doc.get('content', '').lower()
            # Simple keyword matching - check against known disease names
            disease_keywords = [
                ('coffee leaf rust', 'Coffee Leaf Rust'),
                ('red spider mite', 'Red Spider Mites'),
                ('brown eye spot', 'Brown Eye Spot'),
                ('phoma leaf', 'Coffee Phoma Leaf Spot'),
                ('cercospora', 'Cercospora Leaf Spot'),
                ('anthracnose', 'Anthracnose'),
                ('root rot', 'Root Rot / Wilt Disease'),
                ('nitrogen deficiency', 'Nitrogen Deficiency'),
                ('iron deficiency', 'Iron Deficiency'),
                ('potassium deficiency', 'Potassium Deficiency'),
                ('magnesium deficiency', 'Magnesium Deficiency'),
                ('boron deficiency', 'Boron Deficiency'),
                ('zinc deficiency', 'Zinc Deficiency'),
                ('bacterial canker', 'Bacterial Canker'),
                ('coffee berry borer', 'Coffee Berry Borer'),
                ('leaf miner', 'Leaf Miner Damage'),
            ]

            for keyword, disease_name in disease_keywords:
                if keyword in content:
                    diseases_mentioned.add(disease_name)

        # If we found diseases in context, return them
        if diseases_mentioned:
            return list(diseases_mentioned)

        # Otherwise return empty list (use generic questions)
        return []

    def diagnose_web(self, initial_query: str) -> Dict:
        """
        Run diagnosis for web interface (no interactive input)

        Args:
            initial_query: User's initial symptom description

        Returns:
            Dictionary with diagnosis, verification, and state summary
        """
        print(f"\n[QUERY] Initial Query: {initial_query}\n")

        # Initialize state
        self.state_manager.initialize(initial_query)

        # Step 1: Detect ambiguity
        print("[STEP1] Detecting missing information...")
        ambiguity_result = self.ambiguity_detector.detect_ambiguity(initial_query)
        self.state_manager.update_detected_ambiguities(ambiguity_result)

        # Step 2: Retrieve relevant context
        print("[STEP2] Retrieving relevant context...")
        retrieved_docs = self.retriever.retrieve(initial_query, [])

        # Step 3: Evaluate retrieval (CRAG)
        print("[STEP3] Evaluating retrieval relevance...")
        filtered_docs = self.retrieval_evaluator.evaluate_chunks(
            retrieved_docs,
            initial_query
        )
        self.state_manager.update_context(filtered_docs)

        # Step 4: Generate final diagnosis
        print("[STEP4] Generating final diagnosis...")
        final_diagnosis = self.diagnosis_gen.generate_diagnosis(
            initial_query,
            [],  # No user responses yet
            filtered_docs,
            0.7  # Default confidence
        )

        # Step 5: Hallucination check
        print("[STEP5] Verifying diagnosis consistency...")
        verification_report = self.hallucination_checker.check_hallucination(
            initial_query,
            [],
            filtered_docs,
            final_diagnosis.disease_name
        )

        # Compile results
        results = {
            'diagnosis': final_diagnosis,
            'verification': verification_report,
            'state_summary': self.state_manager.get_state_summary(),
            'conversation_history': []
        }

        return results

    def diagnose(self, initial_query: str) -> Dict:
        """
        Run complete multi-turn diagnosis process

        Args:
            initial_query: User's initial symptom description

        Returns:
            Dictionary with diagnosis, conversation history, and verification
        """
        print(f"\n[QUERY] Initial Query: {initial_query}\n")

        # Initialize state and reset priority tracking
        self.state_manager.initialize(initial_query)
        self.clarification_gen.reset_priority_tracking()

        # Main loop
        while True:
            print(f"{'='*60}")
            print(f"Turn {self.state_manager.questions_asked + 1}")
            print(f"Confidence: {self.state_manager.confidence * 100:.1f}%")
            print(f"{'='*60}\n")

            # Step 1: Detect ambiguity
            print("[STEP1] Detecting missing information...")
            ambiguity_result = self.ambiguity_detector.detect_ambiguity(
                self.state_manager.state.initial_query + " " + " ".join(
                    self.state_manager.state.user_responses
                )
            )
            missing_info = ambiguity_result['missing']
            self.state_manager.update_detected_ambiguities(ambiguity_result)

            if missing_info:
                print(f"Missing: {', '.join(missing_info.keys())}")
            else:
                print("All necessary information present")

            # Step 2: Retrieve relevant context
            print("\n[STEP2] Retrieving relevant context...")
            retrieved_docs = self.retriever.retrieve(
                self.state_manager.state.initial_query,
                self.state_manager.state.user_responses
            )

            # Step 3: Evaluate retrieval (CRAG)
            print("[STEP3] Evaluating retrieval relevance...")
            filtered_docs = self.retrieval_evaluator.evaluate_chunks(
                retrieved_docs,
                self.state_manager.state.initial_query
            )
            self.state_manager.update_context(filtered_docs)

            # Step 4: Check stop condition
            if self.state_manager.should_stop():
                print("\n[OK] Stop condition met - Proceeding to diagnosis")
                break

            # Step 5: Generate clarification question (RAC)
            if missing_info or filtered_docs:
                print("\n[STEP4] Generating clarification question...")
                
                # Predict likely diseases to guide question generation
                suspected_diseases = self._predict_likely_diseases(
                    self.state_manager.state.initial_query,
                    filtered_docs
                )
                if suspected_diseases:
                    print(f"[DEBUG] Suspected diseases: {suspected_diseases}")
                
                question = self.clarification_gen.generate_question(
                    self.state_manager.state.initial_query,
                    filtered_docs,
                    self.state_manager.state.user_responses,
                    missing_info,
                    suspected_diseases=suspected_diseases
                )
                print(f"\nSystem: {question}\n")
                self.state_manager.add_system_question(question)

                # Get user response
                user_response = input("You: ").strip()

                if user_response.lower() in ['quit', 'exit', 'done', 'stop']:
                    break

                self.state_manager.add_user_response(user_response)
            else:
                print("No additional information needed")
                break

        # Step 6: Generate final diagnosis
        print("\n" + "="*60)
        print("[STEP5] GENERATING FINAL DIAGNOSIS")
        print("="*60)

        final_diagnosis = self.diagnosis_gen.generate_diagnosis(
            self.state_manager.state.initial_query,
            self.state_manager.state.user_responses,
            self.state_manager.state.retrieved_context,
            self.state_manager.confidence
        )

        # Step 7: Hallucination check
        print("\n[STEP6] Verifying diagnosis consistency...")
        verification_report = self.hallucination_checker.check_hallucination(
            self.state_manager.state.initial_query,
            self.state_manager.state.user_responses,
            self.state_manager.state.retrieved_context,
            final_diagnosis.disease_name
        )

        # Compile results
        results = {
            'diagnosis': final_diagnosis,
            'verification': verification_report,
            'state_summary': self.state_manager.get_state_summary(),
            'conversation_history': self.state_manager.get_conversation_history()
        }

        return results

    def print_results(self, results: Dict) -> None:
        """Print formatted results"""
        diagnosis = results['diagnosis']
        verification = results['verification']

        print(self.diagnosis_gen.format_diagnosis_output(diagnosis))
        print(self.hallucination_checker.format_verification_report(verification))

        if not verification['consistent']:
            print("\n[WARNING] Diagnosis may contain hallucinated information. "
                  "Consult local experts for confirmation.")

    def reset(self) -> None:
        """Reset system for new diagnosis"""
        self.state_manager.reset()

    def start_diagnosis(self, initial_query: str) -> Dict:
        """
        Start multi-turn diagnosis for web interface

        Args:
            initial_query: User's initial symptom description

        Returns:
            Dictionary with 'status' ('question' or 'diagnosis') and question/diagnosis data
        """
        print(f"\n[QUERY] Initial Query: {initial_query}\n")

        if not self._matches_context_or_coffee(initial_query):
            return {
                'status': 'off_topic',
                'message': (
                    "This assistant only diagnoses coffee plant health issues. "
                    "Please describe the symptoms you see on coffee plants (leaf color, spots, wilting, berries, stems, etc.)."
                )
            }

        # Initialize state
        self.state_manager.initialize(initial_query)

        # Step 1: Detect ambiguity
        print("[STEP1] Detecting missing information...")
        ambiguity_result = self.ambiguity_detector.detect_ambiguity(initial_query)
        missing_info = ambiguity_result['missing']
        self.state_manager.update_detected_ambiguities(ambiguity_result)

        # Step 2: Retrieve relevant context
        print("[STEP2] Retrieving relevant context...")
        retrieved_docs = self.retriever.retrieve(initial_query, [])

        # Step 3: Evaluate retrieval (CRAG)
        print("[STEP3] Evaluating retrieval relevance...")
        filtered_docs = self.retrieval_evaluator.evaluate_chunks(
            retrieved_docs,
            initial_query
        )
        self.state_manager.update_context(filtered_docs)

        # Ask first clarification about highest priority missing attribute
        if missing_info:  # Removed filtered_docs requirement - always ask if ambiguity exists
            print("[STEP4] Generating clarification question...")
            # Get first missing attribute to ask about
            missing_keys = list(missing_info.keys())
            primary_missing = missing_keys[0] if missing_keys else None

            question = self.clarification_gen.generate_question(
                initial_query,
                filtered_docs or [],  # Pass empty list if no docs
                [],
                missing_info
            )
            self.state_manager.add_system_question(question, attribute=primary_missing)
            return {
                'status': 'question',
                'question': question
            }
        else:
            # No clarifications needed, go to diagnosis
            return self._generate_final_diagnosis(initial_query, [])

    def submit_answer(self, answer: str) -> Dict:
        """
        Submit user answer and continue multi-turn process

        Args:
            answer: User's answer to the clarification question

        Returns:
            Dictionary with 'status' ('question' or 'diagnosis') and next question/diagnosis
        """
        print(f"\n[USER_ANSWER] {answer}\n")

        if not self._matches_context_or_coffee(answer):
            if self._is_binary_clarification_response(answer):
                # Accept short yes/no-style clarifications when they answer
                # the current question context, and preserve which attribute it
                # referred to for downstream ambiguity resolution.
                self.state_manager.add_user_response(
                    self._tagged_clarification_response(answer)
                )
            # Allow "not sure" / "don't know" type responses as valid (neutral info)
            uncertain_phrases = ['not sure', "don't know", "dont know", "not certain",
                                "can't tell", "cant tell", "unsure", "no idea"]
            if any(phrase in answer.lower() for phrase in uncertain_phrases):
                # Mark this as uncertain but continue with next question
                self.state_manager.add_user_response(f"User uncertain: {answer}")
                print(f"[NOTE] User expressed uncertainty, continuing...\n")
            elif self._is_binary_clarification_response(answer):
                print("[NOTE] Accepted short clarification response, continuing...\n")
            else:
                return {
                    'status': 'off_topic',
                    'message': (
                        "That response doesn't mention coffee plant symptoms. "
                        "Please describe what you observe on the coffee plant (leaves, berries, stems) so I can continue."
                    )
                }
        else:
            # Add user response to state
            self.state_manager.add_user_response(answer)

        # Check stop condition
        if self.state_manager.should_stop():
            print("[OK] Stop condition met")
            return self._generate_final_diagnosis(
                self.state_manager.state.initial_query,
                self.state_manager.state.user_responses
            )

        # Detect remaining ambiguities with updated information
        print("[STEP4] Detecting remaining missing information...")
        full_context = (
            self.state_manager.state.initial_query + " " +
            " ".join(self.state_manager.state.user_responses)
        )
        ambiguity_result = self.ambiguity_detector.detect_ambiguity(full_context)
        missing_info = ambiguity_result['missing']
        self.state_manager.update_detected_ambiguities(ambiguity_result)

        # Filter out attributes we've already asked about
        new_missing = {}
        for key, value in missing_info.items():
            if key not in self.state_manager.state.asked_about_attributes:
                new_missing[key] = value

        print(f"Missing info: {list(missing_info.keys())}, Already asked: {self.state_manager.state.asked_about_attributes}")

        # Retrieve updated context
        retrieved_docs = self.retriever.retrieve(
            self.state_manager.state.initial_query,
            self.state_manager.state.user_responses
        )

        filtered_docs = self.retrieval_evaluator.evaluate_chunks(
            retrieved_docs,
            self.state_manager.state.initial_query
        )
        self.state_manager.update_context(filtered_docs)

        # Ask next question only about NEW missing attributes
        if new_missing:  # Removed filtered_docs requirement - always ask if new ambiguity exists
            print("[STEP5] Generating next clarification question...")
            # Ask about the first new missing attribute
            primary_missing = list(new_missing.keys())[0]

            question = self.clarification_gen.generate_question(
                self.state_manager.state.initial_query,
                filtered_docs or [],  # Pass empty list if no docs
                self.state_manager.state.user_responses,
                new_missing
            )
            self.state_manager.add_system_question(question, attribute=primary_missing)
            return {
                'status': 'question',
                'question': question
            }
        else:
            # All relevant info gathered
            print("[OK] All relevant information collected")
            return self._generate_final_diagnosis(
                self.state_manager.state.initial_query,
                self.state_manager.state.user_responses
            )

    def _generate_final_diagnosis(self, initial_query: str, user_responses: List[str]) -> Dict:
        """
        Generate final diagnosis with hallucination verification

        Args:
            initial_query: Initial user query
            user_responses: List of user responses to clarifications

        Returns:
            Dictionary with 'status', diagnosis, and verification
        """
        print("\n[STEP5] GENERATING FINAL DIAGNOSIS")

        final_diagnosis = self.diagnosis_gen.generate_diagnosis(
            initial_query,
            user_responses,
            self.state_manager.state.retrieved_context,
            self.state_manager.confidence
        )

        print("\n[STEP6] Verifying diagnosis consistency...")
        verification_report = self.hallucination_checker.check_hallucination(
            initial_query,
            user_responses,
            self.state_manager.state.retrieved_context,
            final_diagnosis.disease_name
        )

        # Active gate: ask one more targeted clarification when verification is weak
        if (
            (not verification_report.get('safe_to_finalize', verification_report.get('consistent', True)))
            and self.state_manager.questions_asked < settings.MAX_QUESTIONS
        ):
            full_context = initial_query + " " + " ".join(user_responses)
            ambiguity_result = self.ambiguity_detector.detect_ambiguity(full_context)
            missing_info = ambiguity_result.get('missing', {})
            self.state_manager.update_detected_ambiguities(ambiguity_result)

            new_missing = {
                key: value for key, value in missing_info.items()
                if key not in self.state_manager.state.asked_about_attributes
            }

            if new_missing:
                question = self.clarification_gen.generate_question(
                    initial_query,
                    self.state_manager.state.retrieved_context,
                    user_responses,
                    new_missing
                )
                primary_missing = list(new_missing.keys())[0]
                self.state_manager.add_system_question(question, attribute=primary_missing)
                return {
                    'status': 'question',
                    'question': question
                }

        return {
            'status': 'diagnosis',
            'diagnosis': final_diagnosis,
            'verification': verification_report,
            'state_summary': self.state_manager.get_state_summary(),
            'conversation_history': self._get_conversation_history()
        }

    def _get_conversation_history(self) -> str:
        """
        Get formatted conversation history

        Returns:
            Formatted conversation history string
        """
        return self.state_manager.get_conversation_history()

    def _matches_context_or_coffee(self, text: str) -> bool:
        """
        Determine whether user text is relevant to coffee diagnosis.

        Accepts text that either:
        1. Mentions coffee plant parts or known symptom keywords
        2. Overlaps with retrieved context content (allowing short answers like "white ring")
        """
        if not text:
            return False

        text_lower = text.lower()

        plant_keywords = {
            "coffee",
            "arabica",
            "robusta",
            "plant",
            "tree",
            "stem",
            "branch",
            "berry",
            "berries",
            "bean",
            "beans",
            "crop",
            "cherry",
            "canopy",
        }

        symptom_keywords = {
            "leaf",
            "leaves",
            "spot",
            "spots",
            "patch",
            "patches",
            "color",
            "yellow",
            "brown",
            "red",
            "orange",
            "powder",
            "wilt",
            "wilting",
            "curl",
            "curling",
            "hole",
            "holes",
            "tunnel",
            "dieback",
            "ring",
            "rings",
            "edge",
            "edges",
            "margin",
            "margins",
            "vein",
            "veins",
            "chlorosis",
            "lesion",
            "lesions",
            "scorch",
            "rust",
        }

        if any(keyword in text_lower for keyword in plant_keywords.union(symptom_keywords)):
            return True

        state = getattr(self.state_manager, "state", None)
        if not state or not getattr(state, "retrieved_context", None):
            return False

        words = [w for w in text_lower.split() if len(w) >= 3]
        if not words:
            return False

        for chunk in state.retrieved_context:
            if isinstance(chunk, dict):
                content = chunk.get("content") or ""
            else:
                content = getattr(chunk, "page_content", "") or ""
            content_lower = content.lower()
            if any(word in content_lower for word in words):
                return True

        return False

    def _is_binary_clarification_response(self, text: str) -> bool:
        """Allow short yes/no-style replies for follow-up clarification turns."""
        if not text:
            return False

        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return False

        binary_phrases = {
            "yes", "yeah", "yep", "y", "correct", "true",
            "no", "nope", "nah", "n", "false",
            "yes mostly", "yes partly", "partly", "partially",
            "not really", "a little", "slightly",
        }
        return normalized in binary_phrases

    def _tagged_clarification_response(self, answer: str) -> str:
        """Attach latest asked attribute so brief replies carry usable context."""
        normalized = " ".join((answer or "").strip().split()).lower()
        attrs = self.state_manager.state.asked_about_attributes
        if attrs:
            return f"{attrs[-1]}: {normalized}"
        return normalized
