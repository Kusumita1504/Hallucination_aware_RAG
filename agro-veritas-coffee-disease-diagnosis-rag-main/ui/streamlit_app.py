"""
Streamlit UI Application
Multi-turn Coffee Disease Diagnosis System with Hallucination Detection
"""
import streamlit as st
import os
from pathlib import Path
import sys

# Add project src directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController


@st.cache_resource
def get_controller():
    """Load controller with caching to avoid reinitializing"""
    return CoffeeDiagnosisController(
        data_dir=str(Path(__file__).parent.parent / "data" / "pdfs"),
        vector_db_path=str(Path(__file__).parent.parent / "data" / "vector_db")
    )


def initialize_session():
    """Initialize session state"""
    if 'controller' not in st.session_state:
        with st.spinner("Loading Coffee Disease Diagnosis System... (This takes 2-3 minutes on first run)"):
            st.session_state.controller = get_controller()

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'current_question' not in st.session_state:
        st.session_state.current_question = None

    if 'diagnosis_result' not in st.session_state:
        st.session_state.diagnosis_result = None

    if 'diagnosis_started' not in st.session_state:
        st.session_state.diagnosis_started = False

    if 'awaiting_answer' not in st.session_state:
        st.session_state.awaiting_answer = False


def main():
    # Page configuration
    st.set_page_config(
        page_title="Coffee Disease Diagnosis",
        page_icon="leaf",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .title-style {
            font-size: 2.5rem;
            color: #8B4513;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle-style {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .qa-box {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .diagnosis-box {
            background-color: #e8f4f8;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #0066cc;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ff9800;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(
        """<div class='title-style'>Coffee Disease Diagnosis System</div>""",
        unsafe_allow_html=True
    )
    st.markdown(
        """<div class='subtitle-style'>Multi-turn Assistant with Hallucination-Aware Coffee Disease Diagnosis using RAG</div>""",
        unsafe_allow_html=True
    )

    # Initialize session
    initialize_session()

    # Sidebar information
    with st.sidebar:
        st.header("System Information")

        st.markdown("""
        ### How to Use:
        1. Describe your coffee plant symptoms
        2. Answer clarification questions
        3. Get AI diagnosis with verification
        4. View treatment recommendations

        ### Architecture:
        - **Ambiguity Detection**: Identifies missing information
        - **RAG Retrieval**: Fetches relevant documents
        - **CRAG Filtering**: Evaluates context relevance
        - **RAC Questions**: Grounded clarification queries
        - **Multi-turn Loop**: Iterative refinement
        - **SelfCheckGPT**: Hallucination verification
        """)

        st.divider()

        if st.button("Reset Conversation", key="reset_btn"):
            st.session_state.controller.reset()
            st.session_state.conversation = []
            st.session_state.current_question = None
            st.session_state.diagnosis_result = None
            st.session_state.diagnosis_started = False
            st.session_state.awaiting_answer = False
            st.rerun()

        st.divider()

        st.markdown("""
        **System Status**: OK

        **Knowledge Base**: 3 PDFs loaded
        - Coffee.pdf
        - coffee_cultivation_guide.pdf
        - i4985e.pdf
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Diagnosis Assistant")

        # Display conversation history
        if st.session_state.conversation:
            st.markdown("### Conversation History")
            for i, msg in enumerate(st.session_state.conversation):
                if msg['role'] == 'user':
                    st.markdown(f"**You**: {msg['content']}")
                else:
                    st.markdown(f"**System**: {msg['content']}")
                    if i < len(st.session_state.conversation) - 1:
                        st.divider()

        # Diagnosis result display
        if st.session_state.diagnosis_result:
            # Check if we got a diagnosis or another question (active gating)
            if st.session_state.diagnosis_result.get('status') == 'question':
                # Active gating triggered - we got another question instead of diagnosis
                st.session_state.current_question = st.session_state.diagnosis_result['question']
                st.session_state.awaiting_answer = True
                st.session_state.diagnosis_result = None
                st.session_state.gating_triggered = True  # Set flag to show message on next render
                st.rerun()
            elif 'diagnosis' in st.session_state.diagnosis_result:
                st.markdown("---")
                st.subheader("Diagnosis Result")

                diagnosis = st.session_state.diagnosis_result['diagnosis']
                verification = st.session_state.diagnosis_result['verification']

                # Disease diagnosis
                col_disease, col_conf = st.columns(2)
                with col_disease:
                    st.markdown(f"### Disease")
                    st.write(f"**{diagnosis.disease_name}**")

                with col_conf:
                    st.markdown(f"### Confidence")
                    confidence_pct = diagnosis.confidence * 100
                    st.metric("Score", f"{confidence_pct:.1f}%")

                # Details
                st.markdown("#### Reason")
                st.write(diagnosis.reason)

                st.markdown("#### Treatment")
                st.write(diagnosis.treatment)

                st.markdown("#### Prevention")
                st.write(diagnosis.prevention)

                st.markdown("#### Source")
                st.write(diagnosis.source)

                # RAG evidence for the final decision
                ctx_state = getattr(st.session_state.controller, "state_manager", None)
                ctx = ctx_state.state.retrieved_context if ctx_state and getattr(ctx_state, "state", None) else []
                if ctx:
                    st.markdown("#### Evidence (RAG sources)")
                    for i, chunk in enumerate(ctx[:5], 1):
                        source = chunk.get("source", "Unknown")
                        source_type = chunk.get("source_type", "PDF")  # Default to PDF if not specified
                        page = chunk.get("metadata", {}).get("page")
                        snippet = chunk.get("content", "")[:400]
                        page_info = f" (page {page + 1})" if isinstance(page, int) else ""
                        
                        # Add source type tag
                        source_tag = f"[{source_type}]" if source_type else ""
                        st.markdown(f"**{i}. {source_tag} {source}{page_info}**")
                        st.caption(snippet + ("..." if len(snippet) == 400 else ""))

                # Verification
                st.markdown("---")
                st.subheader("Hallucination Verification")

                if verification['consistent']:
                    status = "[OK] Verified"
                    status_color = "green"
                else:
                    status = "[WARNING] Issues Detected"
                    status_color = "orange"

                st.markdown(f"<div style='color: {status_color}; font-size: 1.2rem;'><b>{status}</b></div>",
                           unsafe_allow_html=True)

                st.write(f"**Consistency Score**: {verification['consistency_score'] * 100:.1f}%")
                st.write(f"**Agreement Level**: {verification['agreement']}")
                st.write(f"**Generations Checked**: {verification['num_generations']}")

                if verification['warnings']:
                    st.markdown("##### Warnings:")
                    for warning in verification['warnings']:
                        st.warning(warning)

                # New diagnosis button
                if st.button("Start New Diagnosis", key="new_diagnosis_btn"):
                    st.session_state.controller.reset()
                    st.session_state.conversation = []
                    st.session_state.current_question = None
                    st.session_state.diagnosis_result = None
                    st.session_state.diagnosis_started = False
                    st.session_state.awaiting_answer = False
                    st.rerun()

        else:
            # Initial input or multi-turn questions
            if not st.session_state.diagnosis_started:
                # Initial symptom input
                st.markdown("### Describe Your Coffee Plant Symptoms")

                user_input = st.text_area(
                    "Enter your observation",
                    placeholder="E.g., 'Leaves are turning yellow with brown spots on the edges'",
                    height=100,
                    label_visibility="collapsed"
                )

                if st.button("Submit", key="submit_initial"):
                    if user_input.strip():
                        # Start diagnosis process
                        with st.spinner("Analyzing symptoms and checking for missing information..."):
                            try:
                                result = st.session_state.controller.start_diagnosis(user_input)
                                status = result.get('status')
                                st.session_state.conversation.append({
                                    'role': 'user',
                                    'content': user_input
                                })

                                if status == 'off_topic':
                                    st.warning(result['message'])
                                    st.session_state.conversation.append({
                                        'role': 'assistant',
                                        'content': result['message']
                                    })
                                else:
                                    st.session_state.diagnosis_started = True

                                    if status == 'question':
                                        # System is asking a clarification question
                                        st.session_state.current_question = result['question']
                                        st.session_state.awaiting_answer = True
                                        st.session_state.conversation.append({
                                            'role': 'assistant',
                                            'content': result['question']
                                        })
                                    elif status == 'diagnosis':
                                        # Got diagnosis directly (no clarifications needed)
                                        st.session_state.diagnosis_result = result

                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error during diagnosis: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())
                    else:
                        st.warning("Please describe your symptoms first.")

            elif st.session_state.awaiting_answer:
                # Display current question and get answer
                st.markdown("### Answer the Question")
                
                # Show notification if active gating was triggered
                if st.session_state.get('gating_triggered', False):
                    st.info("⚠️ **Verification Alert**: The initial diagnosis had low confidence. Asking one more targeted question to improve accuracy.")
                    st.session_state.gating_triggered = False  # Clear flag after showing
                
                st.info(f"**Question**: {st.session_state.current_question}")

                # Manage answer input state and clearing between runs
                if "answer_input" not in st.session_state:
                    st.session_state.answer_input = ""
                if st.session_state.get("clear_answer_input"):
                    st.session_state.answer_input = ""
                    st.session_state.clear_answer_input = False

                user_answer = st.text_area(
                    "Your answer:",
                    placeholder="Provide details based on the question",
                    height=100,
                    label_visibility="collapsed",
                    key="answer_input",
                    value=st.session_state.answer_input
                )

                col_submit, col_skip = st.columns(2)

                with col_submit:
                    if st.button("Submit Answer", key="submit_answer"):
                        if user_answer.strip():
                            with st.spinner("Processing your answer..."):
                                try:
                                    result = st.session_state.controller.submit_answer(user_answer)
                                    st.session_state.conversation.append({
                                        'role': 'user',
                                        'content': user_answer
                                    })
                                    status = result.get('status')

                                    if status == 'off_topic':
                                        st.warning(result['message'])
                                        st.session_state.conversation.append({
                                            'role': 'assistant',
                                            'content': result['message']
                                        })
                                    else:
                                        # Flag to clear input on next render
                                        st.session_state.clear_answer_input = True

                                        if status == 'question':
                                            # Another clarification question
                                            st.session_state.current_question = result['question']
                                            st.session_state.conversation.append({
                                                'role': 'assistant',
                                                'content': result['question']
                                            })
                                        elif status == 'diagnosis':
                                            # Got diagnosis
                                            st.session_state.diagnosis_result = result
                                            st.session_state.awaiting_answer = False

                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error processing answer: {str(e)}")
                                    import traceback
                                    st.error(traceback.format_exc())
                        else:
                            st.warning("Please provide an answer.")

                with col_skip:
                    if st.button("Skip to Diagnosis", key="skip_question"):
                        with st.spinner("Generating diagnosis with current information..."):
                            try:
                                result = st.session_state.controller._generate_final_diagnosis(
                                    st.session_state.controller.state_manager.state.initial_query,
                                    st.session_state.controller.state_manager.state.user_responses
                                )
                                st.session_state.diagnosis_result = result
                                st.session_state.awaiting_answer = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating diagnosis: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())

    with col2:
        st.subheader("Diagnostic Info")

        if st.session_state.diagnosis_started and st.session_state.controller.state_manager:
            state = st.session_state.controller.state_manager.get_state_summary()

            st.metric("Questions Asked", state['questions_asked'])
            st.metric("Confidence Level", f"{state['confidence'] * 100:.1f}%")
            st.metric("Ambiguities", len(state['ambiguities'].get('missing', {})))

            if state['possible_diseases']:
                st.markdown("**Possible Diseases:**")
                for disease in state['possible_diseases']:
                    st.write(f"- {disease}")

        st.markdown("---")

        if st.session_state.awaiting_answer:
            st.info(f"Confidence: {st.session_state.controller.state_manager.confidence * 100:.1f}%")

        st.subheader("Tips")
        st.markdown("""
        - Be specific about symptoms (colors, patterns)
        - Describe affected plant parts (leaves, stems, fruits)
        - Mention when symptoms appeared
        - Answer clarification questions
        """)


if __name__ == "__main__":
    main()
