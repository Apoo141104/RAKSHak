import os
import sys
import torch
import streamlit as st
from dotenv import load_dotenv

# Presidio
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider, SpacyNlpEngine

# NLP and HuggingFace
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Groq API
from groq import Groq


# --- Environment Variable Setup ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Please set it in .env or Streamlit secrets.")
        st.stop()

# Force CPU-only and prevent CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Custom CSS for Enhanced UI ---
st.markdown("""
<style>
    /* Main Container */
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Headers */
    .header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Text Areas */
    .stTextArea>textarea {
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Tabs */
    .stTabs [role="tablist"] {
        margin-bottom: 1rem;
    }
    
    /* Custom Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 10px;
    }
    
    .badge-primary {
        color: #fff;
        background-color: #3498db;
    }
    
    .badge-warning {
        color: #212529;
        background-color: #ffc107;
    }
    
    .badge-danger {
        color: #fff;
        background-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# --- Analyzer Functions ---

@st.cache_resource
def get_presidio_analyzer():
    # --- Load Spacy model manually ---
    import spacy
    try:
        nlp = spacy.load("en_core_web_lg")
    except:
        from spacy.cli import download
        download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
    
    # --- Configure NLP engine for Presidio ---
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
    })
    nlp_engine = provider.create_engine()

    # --- Create AnalyzerEngine with custom recognizers ---
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

    # Add custom regex patterns
    from presidio_analyzer import PatternRecognizer, Pattern
    custom_patterns = {
        "EMPLOYEE_ID": r"E\d{2}[A-Z]{2,4}U\d{4}",
        "IN_AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "IN_PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "IN_PASSPORT": r"\b[A-Z][0-9]{7}\b",
        "IN_VOTER": r"\b[A-Z]{3}[0-9]{7}\b",
        "IN_VEHICLE_REGISTRATION": r"\b[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}\b"
    }
    for entity, pattern in custom_patterns.items():
        recognizer = PatternRecognizer(
            supported_entity=entity,
            patterns=[Pattern(name=entity, regex=pattern, score=0.9)]
        )
        analyzer.registry.add_recognizer(recognizer)
    
    return analyzer


# Get the analyzer instance
analyzer = get_presidio_analyzer()

@st.cache_resource
def get_ner_pipeline():
    """
    Initializes and returns a HuggingFace NER pipeline.
    Uses a smaller model ('dslim/bert-base-NER') for Streamlit Cloud deployment
    and a larger one ('ai4bharat/IndicNer') for local development for better accuracy.
    This function is cached.
    """
    try:
        # Check if running on Streamlit Cloud (heuristic: 'streamlit' in sys.modules)
        # and prioritize a smaller model for efficiency.
        # Otherwise, use a potentially larger, more accurate model.
        if 'streamlit' in sys.modules:
            model_name = "dslim/bert-base-NER"
        else:
            # Fallback for local development or if 'streamlit' isn't in sys.modules
            # and a larger model is preferred/possible.
            # Make sure ai4bharat/IndicNer is installed or available if chosen.
            model_name = "dslim/bert-base-NER" # Changed to bert-base-NER for broader compatibility
            # original: "ai4bharat/IndicNer"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # Ensure the pipeline runs on CPU explicitly
        return pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # Force CPU by setting device to -1
            aggregation_strategy="simple" # Aggregates sub-word tokens into single entities
        )
    except Exception as e:
        st.error(f"NER Model Error: {str(e)}")
        st.info("Please ensure you have the required models downloaded or try running locally first.")
        return None

# Get the NER pipeline instance
ner_pipe = get_ner_pipeline()

@st.cache_resource
def get_sensitivity_model():
    """
    Initializes and returns a zero-shot classification pipeline for sensitivity analysis.
    This uses a pre-trained model to classify text sensitivity.
    This function is cached.
    """
    try:
        # Determine device: GPU (0) if available, otherwise CPU (-1)
        device_id = 0 if torch.cuda.is_available() else -1
        sensitivity_pipe = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli", 
            device=device_id
        )
        return sensitivity_pipe
    except Exception as e:
        st.error(f"Failed to load sensitivity model: {str(e)}")
        st.info("Ensure 'facebook/bart-large-mnli' model is accessible or try again.")
        return None

# Get the sensitivity model instance
sensitivity_model = get_sensitivity_model()

# --- Helper Functions ---
def is_sensitive(text):
    """
    Checks if the given text is sensitive using the pre-trained sensitivity model.
    Returns True if classified as 'confidential', 'sensitive', or 'personal information'
    with a confidence score above 0.7.
    """
    if not sensitivity_model:
        return False # Cannot perform sensitivity check if model failed to load
    
    candidate_labels = ["confidential", "public", "sensitive", "personal information"]
    try:
        # Perform zero-shot classification
        prediction = sensitivity_model(text, candidate_labels)
        top_label = prediction["labels"][0]
        top_score = prediction["scores"][0]
        
        # Define sensitivity based on top label and score threshold
        return top_label.lower() in ["confidential", "sensitive", "personal information"] and top_score > 0.7
    except Exception as e:
        st.error(f"Sensitivity check failed: {str(e)}")
        return False

# Mapping for NER entity tags for cleaner display
NER_TAGS = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION", "MISC": "MISC"}

def redact_ner(text):
    """
    Redacts named entities (PERSON, LOCATION, ORGANIZATION, MISC) from text
    using the NER pipeline. Replaces detected entities with a placeholder tag.
    Returns the redacted text and the detected entities.
    """
    if not ner_pipe:
        return text, [] # Return original text if NER pipeline failed to load

    results = ner_pipe(text)
    redacted = list(text) # Convert to list for mutable character replacement
    entity_counts = {} # To keep track of multiple occurrences of the same entity type

    # Iterate through results in reverse order to avoid index shifting issues during redaction
    for ent in sorted(results, key=lambda x: x['start'], reverse=True):
        entity_type = NER_TAGS.get(ent['entity_group'], ent['entity_group'])
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        tag = f"[{entity_type}_{entity_counts[entity_type]}]"
        # Replace the original entity text with the placeholder tag
        redacted[ent['start']:ent['end']] = list(tag)
    
    return "".join(redacted), results

def redact_presidio(text):
    """
    Redacts PII based on predefined patterns using Presidio Analyzer.
    Returns the redacted text and the detected entities.
    """
    # Get all supported entity types from the Presidio analyzer's registry
    all_supported_entities = set()
    for recognizer in analyzer.registry.recognizers:
        all_supported_entities.update(recognizer.supported_entities)
    entities_to_analyze = list(all_supported_entities)
    
    # Analyze the text for PII entities
    results = analyzer.analyze(text=text, entities=entities_to_analyze, language="en")
    redacted = list(text) # Convert to list for mutable character replacement
    entity_counts = {} # To keep track of multiple occurrences of the same entity type

    # Iterate through results in reverse order to avoid index shifting issues
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        entity_type = r.entity_type
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        tag = f"[{entity_type}_{entity_counts[entity_type]}]"
        # Replace the original PII text with the placeholder tag
        redacted[r.start:r.end] = list(tag)
    
    return "".join(redacted), results

def generate_llm_answer(prompt):
    """
    Generates a response from the Groq LLM (Llama3-8B-8192 model) based on the given prompt.
    """
    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 # Controls randomness of the response
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error] Failed to get response from LLM: {str(e)}"

# --- Enhanced UI Layout ---
def main():
    """
    Main function to set up the Streamlit application UI and logic.
    """
    # Set page configuration for better layout and appearance
    st.set_page_config(
        page_title="RAKSHAK: Secure PII Redaction", 
        page_icon="🛡️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables if they don't exist
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    
    # Sidebar for settings and information
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        # Placeholder for future analysis mode selection (currently not implemented)
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Standard", "Deep Scan"], # Deep Scan could imply more models or broader entity types
            help="Choose between standard or more thorough analysis (future feature)"
        )
        
        st.markdown("## 🔍 Recognized Patterns")
        with st.expander("View All Patterns"):
            # Display all entity types recognized by Presidio
            all_display_entities = set()
            for recognizer in analyzer.registry.recognizers:
                all_display_entities.update(recognizer.supported_entities)
            for entity in sorted(all_display_entities):
                st.markdown(f"- `{entity}`")
        
        st.markdown("---")
        st.markdown("""
        **About RAKSHAK** RAKSHAK (meaning 'protector' in Hindi) is a secure PII redaction system 
        that combines:  
        - **Presidio**: For pattern-based PII detection (e.g., Aadhar, PAN, Passport).  
        - **HuggingFace NER Models**: For Named Entity Recognition (e.g., PERSON, LOCATION, ORGANIZATION).  
        - **Sensitivity Classification**: To assess overall text sensitivity before LLM interaction.  
        - **Secure LLM Integration**: Uses Groq's API with redacted text to ensure privacy.
        """)
    
    # Main content area header
    st.markdown("<h1 class='header'>🛡️ RAKSHAK: Secure PII Redaction</h1>", unsafe_allow_html=True)
    st.markdown("Automated content analysis and redaction for sensitive information.")
    
    # Input card for user text
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ✏️ Text Input")
        user_input = st.text_area(
            "Enter text to analyze:",
            height=200,
            key="user_input_area", # Unique key for text area
            label_visibility="collapsed", # Hides default label
            placeholder="Paste or type sensitive content here (e.g., 'My name is John Doe, my Aadhar is 1234 5678 9012, and I work at Google.')"
        )
        
        col1, col2 = st.columns([1, 3]) # Layout columns for buttons
        with col1:
            # Analyze & Redact button
            if st.button("🔍 Analyze & Redact", key="analyze_button", type="primary"):
                if not user_input.strip():
                    st.warning("Please enter some text to analyze.")
                else:
                    st.session_state.show_results = True # Flag to show results section
                    with st.spinner("Analyzing content..."):
                        # Perform Presidio redaction first
                        presidio_redacted, presidio_entities = redact_presidio(user_input)
                        # Then perform NER redaction on the Presidio-redacted text
                        ner_redacted, ner_entities = redact_ner(presidio_redacted)
                        
                        # Store results in session state for later display
                        st.session_state.presidio_entities = presidio_entities
                        st.session_state.ner_entities = ner_entities
                        st.session_state.redacted_text = ner_redacted
                        
                        # Generate LLM response based on sensitivity
                        with st.spinner("Consulting AI assistant..."):
                            if is_sensitive(user_input):
                                st.session_state.sensitive_warning = True
                                # Send redacted text to LLM if original is sensitive
                                llm_output = generate_llm_answer(ner_redacted) 
                            else:
                                st.session_state.sensitive_warning = False
                                # Can send original or redacted, but for consistency, still send redacted
                                llm_output = generate_llm_answer(ner_redacted) 
                            st.session_state.llm_output = llm_output
                        
                        # Add current analysis to history
                        st.session_state.history.append(
                            (user_input, ner_redacted, llm_output)
                        )
        with col2:
            # New Analysis button to clear and restart
            if st.session_state.show_results and st.button("🔄 New Analysis", key="new_question_button"):
                st.session_state.show_results = False
                st.session_state.user_input_area = "" # Clear text area
                st.experimental_rerun() # Rerun the app to clear displayed results
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Results display section (only shown after analysis)
    if st.session_state.show_results:
        st.markdown("---") # Separator
        # Tabs for different result views
        tab1, tab2, tab3 = st.tabs(["🔍 Detection Results", "🧼 Redacted Text", "🤖 AI Response"])
        
        with tab1:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                col1_res, col2_res = st.columns(2) # Columns for Presidio and NER results
                
                with col1_res:
                    st.markdown("### Presidio Pattern Matches")
                    # Display Presidio detected entities
                    if hasattr(st.session_state, 'presidio_entities') and st.session_state.presidio_entities:
                        for entity in st.session_state.presidio_entities:
                            st.markdown(f"""
                            - <span class='badge badge-primary'>{entity.entity_type}</span>: 
                            `{user_input[entity.start:entity.end]}` (pos: {entity.start}-{entity.end})
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("✅ No pattern matches found by Presidio.")
                
                with col2_res:
                    st.markdown("### NER Model Detections")
                    # Display NER detected entities
                    if hasattr(st.session_state, 'ner_entities') and st.session_state.ner_entities:
                        for ent in st.session_state.ner_entities:
                            etype = NER_TAGS.get(ent['entity_group'], ent['entity_group'])
                            st.markdown(f"""
                            - <span class='badge badge-warning'>{etype}</span>: 
                            `{ent['word']}` (pos: {ent['start']}-{ent['end']})
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("✅ No entities detected by NER model.")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Redacted Output")
                # Display the final redacted text
                st.code(st.session_state.redacted_text, language="text")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### AI Assistant Response")
                # Display sensitive content warning if applicable
                if hasattr(st.session_state, 'sensitive_warning') and st.session_state.sensitive_warning:
                    st.warning("⚠️ Sensitive content detected - AI query was made using redacted text for privacy.")
                # Display the LLM's response
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px;'>
                    {st.session_state.llm_output}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    
    # History section, collapsed by default
    if st.session_state.history:
        st.markdown("---")
        with st.expander("📚 Analysis History", expanded=False):
            # Iterate through stored history items
            for i, (original, redacted, llm_resp) in enumerate(st.session_state.history):
                with st.container():
                    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"#### Analysis #{i+1}")
                    
                    # Tabs for each history entry
                    hist_tab1, hist_tab2, hist_tab3 = st.tabs(["Original Text", "Redacted Text", "AI Response"])
                    
                    with hist_tab1:
                        st.code(original, language="text")
                    
                    with hist_tab2:
                        st.code(redacted, language="text")
                    
                    with hist_tab3:
                        st.markdown(f"""
                        <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px;'>
                            {llm_resp}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True) # Close card for history item

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
