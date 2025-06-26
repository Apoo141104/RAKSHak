import os
import torch
import streamlit as st
from dotenv import load_dotenv
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
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

# --- Analyzer Functions (Same as before) ---
@st.cache_resource
def get_presidio_analyzer():
    analyzer = AnalyzerEngine()
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

analyzer = get_presidio_analyzer()

@st.cache_resource
def get_ner_pipeline():
    model_name = "ai4bharat/IndicNer"
    device_id = 0 if torch.cuda.is_available() else -1
    model = AutoModelForTokenClassification.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_pipe = pipeline(
        "ner", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="simple", 
        device=device_id
    )
    return ner_pipe

ner_pipe = get_ner_pipeline()

@st.cache_resource
def get_sensitivity_model():
    try:
        device_id = 0 if torch.cuda.is_available() else -1
        sensitivity_pipe = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli", 
            device=device_id
        )
        return sensitivity_pipe
    except Exception as e:
        st.error(f"Failed to load sensitivity model: {str(e)}")
        return None

sensitivity_model = get_sensitivity_model()

# --- Helper Functions (Same as before) ---
def is_sensitive(text):
    if not sensitivity_model:
        return False
    candidate_labels = ["confidential", "public", "sensitive", "personal information"]
    try:
        prediction = sensitivity_model(text, candidate_labels)
        top_label = prediction["labels"][0]
        top_score = prediction["scores"][0]
        return top_label.lower() in ["confidential", "sensitive", "personal information"] and top_score > 0.7
    except Exception as e:
        st.error(f"Sensitivity check failed: {str(e)}")
        return False

NER_TAGS = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION", "MISC": "MISC"}

def redact_ner(text):
    results = ner_pipe(text)
    redacted = list(text)
    entity_counts = {}
    for ent in sorted(results, key=lambda x: x['start'], reverse=True):
        entity_type = NER_TAGS.get(ent['entity_group'], ent['entity_group'])
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        tag = f"[{entity_type}_{entity_counts[entity_type]}]"
        redacted[ent['start']:ent['end']] = list(tag)
    return "".join(redacted), results

def redact_presidio(text):
    all_supported_entities = set()
    for recognizer in analyzer.registry.recognizers:
        all_supported_entities.update(recognizer.supported_entities)
    entities = list(all_supported_entities)
    
    results = analyzer.analyze(text=text, entities=entities, language="en")
    redacted = list(text)
    entity_counts = {}
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        entity_type = r.entity_type
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        tag = f"[{entity_type}_{entity_counts[entity_type]}]"
        redacted[r.start:r.end] = list(tag)
    return "".join(redacted), results

def generate_llm_answer(prompt):
    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error] {str(e)}"

# --- Enhanced UI Layout ---
def main():
    st.set_page_config(
        page_title="RAKSHAK", 
        page_icon="🛡️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    
    # Sidebar with settings
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Standard", "Deep Scan"],
            help="Choose between standard or more thorough analysis"
        )
        
        st.markdown("## 🔍 Recognized Patterns")
        with st.expander("View All Patterns"):
            all_display_entities = set()
            for recognizer in analyzer.registry.recognizers:
                all_display_entities.update(recognizer.supported_entities)
            for entity in sorted(all_display_entities):
                st.markdown(f"- `{entity}`")
        
        st.markdown("---")
        st.markdown("""
        **About This Tool**  
        A secure PII redaction system combining:  
        - Presidio pattern matching  
        - NER model detection  
        - Sensitivity classification  
        - Secure LLM integration
        """)
    
    # Main content area
    st.markdown("<h1 class='header'>🛡️ RAKSHAK</h1>", unsafe_allow_html=True)
    
    # Input card
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ✏️ Text Input")
        user_input = st.text_area(
            "Enter text to analyze:",
            height=200,
            key="user_input_area",
            label_visibility="collapsed",
            placeholder="Paste or type sensitive content here..."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔍 Analyze & Redact", key="analyze_button", type="primary"):
                if not user_input.strip():
                    st.warning("Please enter some text to analyze.")
                else:
                    st.session_state.show_results = True
                    with st.spinner("Analyzing content..."):
                        # Perform analysis
                        presidio_redacted, presidio_entities = redact_presidio(user_input)
                        ner_redacted, ner_entities = redact_ner(presidio_redacted)
                        
                        # Store results
                        st.session_state.presidio_entities = presidio_entities
                        st.session_state.ner_entities = ner_entities
                        st.session_state.redacted_text = ner_redacted
                        
                        # Generate LLM response
                        with st.spinner("Consulting AI assistant..."):
                            if is_sensitive(user_input):
                                st.session_state.sensitive_warning = True
                                llm_output = generate_llm_answer(ner_redacted)
                            else:
                                st.session_state.sensitive_warning = False
                                llm_output = generate_llm_answer(ner_redacted)
                            st.session_state.llm_output = llm_output
                        
                        # Store in history
                        st.session_state.history.append(
                            (user_input, ner_redacted, llm_output)
                        )
        with col2:
            if st.session_state.show_results and st.button("🔄 New Analysis", key="new_question_button"):
                st.session_state.show_results = False
                st.session_state.user_input_area = ""
                st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Results display
    if st.session_state.show_results:
        # Results tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Detection Results", "🧼 Redacted Text", "🤖 AI Response"])
        
        with tab1:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Presidio Pattern Matches")
                    if hasattr(st.session_state, 'presidio_entities') and st.session_state.presidio_entities:
                        for entity in st.session_state.presidio_entities:
                            st.markdown(f"""
                            - <span class='badge badge-primary'>{entity.entity_type}</span>: 
                            `{user_input[entity.start:entity.end]}` (pos: {entity.start}-{entity.end})
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("✅ No pattern matches found")
                
                with col2:
                    st.markdown("### NER Model Detections")
                    if hasattr(st.session_state, 'ner_entities') and st.session_state.ner_entities:
                        for ent in st.session_state.ner_entities:
                            etype = NER_TAGS.get(ent['entity_group'], ent['entity_group'])
                            st.markdown(f"""
                            - <span class='badge badge-warning'>{etype}</span>: 
                            `{ent['word']}` (pos: {ent['start']}-{ent['end']})
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("✅ No entities detected")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Redacted Output")
                st.code(st.session_state.redacted_text, language="text")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### AI Assistant Response")
                if hasattr(st.session_state, 'sensitive_warning') and st.session_state.sensitive_warning:
                    st.warning("⚠️ Sensitive content detected - used redacted text for AI query")
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px;'>
                    {st.session_state.llm_output}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    
    # History section
    if st.session_state.history:
        with st.expander("📚 Analysis History", expanded=False):
            for i, (original, redacted, llm_resp) in enumerate(st.session_state.history):
                with st.container():
                    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"#### Analysis #{i+1}")
                    
                    hist_tab1, hist_tab2, hist_tab3 = st.tabs(["Original", "Redacted", "AI Response"])
                    
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
                    
                    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()