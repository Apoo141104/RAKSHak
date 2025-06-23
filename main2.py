import os
import torch
import streamlit as st
from dotenv import load_dotenv # For local development
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from groq import Groq

# --- Environment Variable / Streamlit Secrets Handling ---
# In Streamlit Cloud, you configure secrets via the dashboard.
# Locally, you can use a .env file.
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    # Fallback for local development if not using Streamlit secrets
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Please set it in .env or Streamlit secrets.")
        st.stop()


# --- Setup Presidio Analyzer ---
@st.cache_resource
def get_presidio_analyzer():
    """Initializes and returns the Presidio AnalyzerEngine with custom patterns."""
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
        recognizer = PatternRecognizer(supported_entity=entity, patterns=[Pattern(name=entity, regex=pattern, score=0.9)])
        analyzer.registry.add_recognizer(recognizer)
    return analyzer

analyzer = get_presidio_analyzer()

# --- Load HuggingFace NER model (IndicNER) ---
@st.cache_resource
def get_ner_pipeline():
    """Loads and caches the HuggingFace NER pipeline."""
    model_name = "ai4bharat/IndicNer"
    # Determine the device: Use GPU (cuda) if available, otherwise CPU
    # Ensure this is compatible with the `pipeline` device parameter (0 for GPU, -1 for CPU)
    device_id = 0 if torch.cuda.is_available() else -1

    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create NER pipeline, moving model to appropriate device
    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device_id)
    return ner_pipe

ner_pipe = get_ner_pipeline()

# --- Load Zero-shot classification model for sensitivity ---
@st.cache_resource
def get_sensitivity_model():
    """Loads and caches the zero-shot classification model for sensitivity."""
    try:
        # Determine the device: Use GPU (cuda) if available, otherwise CPU
        device_id = 0 if torch.cuda.is_available() else -1
        sensitivity_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)
        return sensitivity_pipe
    except Exception as e:
        st.error(f"Failed to load sensitivity model: {str(e)}")
        return None

sensitivity_model = get_sensitivity_model()

def is_sensitive(text):
    """
    Checks if the given text is sensitive using a zero-shot classification model.
    Returns True if classified as confidential, sensitive, or personal information with high confidence.
    """
    if not sensitivity_model:
        return False
    candidate_labels = ["confidential", "public", "sensitive", "personal information"]
    try:
        prediction = sensitivity_model(text, candidate_labels)
        top_label = prediction["labels"][0]
        top_score = prediction["scores"][0]
        # Define sensitivity based on top label and a confidence threshold
        return top_label.lower() in ["confidential", "sensitive", "personal information"] and top_score > 0.7
    except Exception as e:
        st.error(f"Sensitivity check failed: {str(e)}")
        return False

# Mapping for common NER tags to more descriptive names
NER_TAGS = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION", "MISC": "MISC"}

def redact_ner(text):
    """
    Redacts entities detected by the HuggingFace NER pipeline.
    Replaces detected entities with tags like [ENTITY_TYPE_N].
    """
    results = ner_pipe(text)
    redacted = list(text) # Convert to list for mutable manipulation
    entity_counts = {}
    # Sort results by start position in reverse order to avoid index shifting issues during redaction
    for ent in sorted(results, key=lambda x: x['start'], reverse=True):
        entity_type = NER_TAGS.get(ent['entity_group'], ent['entity_group'])
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        tag = f"[{entity_type}_{entity_counts[entity_type]}]"
        
        # Replace the detected entity span with the tag
        redacted[ent['start']:ent['end']] = list(tag)
    
    return "".join(redacted), results

def redact_presidio(text):
    """
    Redacts entities detected by Presidio Analyzer using predefined patterns.
    Replaces detected entities with tags like [ENTITY_TYPE_N].
    """
    # FIX: Correctly get all supported entities from the analyzer registry
    # Collect all supported entities from all recognizers in the registry
    all_supported_entities = set()
    for recognizer in analyzer.registry.recognizers:
        all_supported_entities.update(recognizer.supported_entities)
    
    entities = list(all_supported_entities)

    results = analyzer.analyze(text=text, entities=entities, language="en")
    
    redacted = list(text) # Convert to list for mutable manipulation
    entity_counts = {}
    # Sort results by start position in reverse order to avoid index shifting issues during redaction
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        entity_type = r.entity_type
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        tag = f"[{entity_type}_{entity_counts[entity_type]}]"
        
        # Replace the detected entity span with the tag
        redacted[r.start:r.end] = list(tag)
        
    return "".join(redacted), results

def generate_llm_answer(prompt):
    """
    Generates a response from the Groq LLM.
    """
    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama3-8b-8192", # Or another suitable model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 # Optional: adjust for creativity
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error] {str(e)}"

# --- Streamlit UI ---
st.set_page_config(page_title="PII Redactor", layout="wide")
st.title("🛡️ Unified PII Protection: Regex + NER + Sensitivity Detection")

# Initialize session history for storing interactions
if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("📦 Default Presidio Recognizers")
# Display built-in Presidio entities for user reference
# Also update this to correctly reflect how entities are retrieved
all_display_entities = set()
for recognizer in analyzer.registry.recognizers:
    all_display_entities.update(recognizer.supported_entities)
st.code(", ".join(sorted(all_display_entities)))


user_input = st.text_area("✏️ Enter your text here:", height=200, key="user_input_area")

if st.button("🔐 Analyze & Redact", key="analyze_button"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Perform Presidio redaction first
        presidio_redacted, presidio_entities = redact_presidio(user_input)
        # Then perform NER redaction on the text already processed by Presidio
        ner_redacted, ner_entities = redact_ner(presidio_redacted)

        st.subheader("🔍 Detected Presidio Entities")
        if presidio_entities:
            for entity in presidio_entities:
                # Display original span for Presidio entities
                st.write(f"- {entity.entity_type}: {user_input[entity.start:entity.end]} (pos: {entity.start}-{entity.end})")
        else:
            st.write("✅ No entities detected by Presidio")

        st.subheader("🔍 Detected HuggingFace NER Entities")
        if ner_entities:
            for ent in ner_entities:
                etype = NER_TAGS.get(ent['entity_group'], ent['entity_group'])
                # Display original word from NER (note: this might be from the presidio-redacted text)
                st.write(f"- {etype}: {ent['word']} (pos: {ent['start']}-{ent['end']})")
        else:
            st.write("✅ No entities detected by HuggingFace NER")

        st.subheader("🧼 Final Redacted Text")
        st.code(ner_redacted)

        st.subheader("🤖 LLM (Groq) Answer")
        # Check sensitivity on original user input
        if is_sensitive(user_input):
            st.warning("⚠️ Sensitive content detected in original input. Sending redacted text to LLM.")
            llm_output = generate_llm_answer(ner_redacted) # Always send redacted text to LLM
        else:
            llm_output = generate_llm_answer(ner_redacted) # Always send redacted text to LLM
        st.text_area("LLM Response:", value=llm_output, height=150, key="llm_response_area")
        
        # Store the interaction in history
        st.session_state.history.append((user_input, ner_redacted, llm_output))

# Button to clear input and allow a new question
if st.button("🔁 Ask Another Question", key="rerun_button"):
    st.session_state.user_input_area = "" # Clear the text area
    st.session_state.history = [] # Clear history on new question (optional)
    st.rerun()

# Display interaction history
if st.session_state.history:
    st.subheader("📚 Interaction History")
    for i, (original, redacted, llm_resp) in enumerate(st.session_state.history):
        with st.expander(f"Interaction {i+1}"):
            st.write("**Original Text:**")
            st.code(original)
            st.write("**Redacted Text:**")
            st.code(redacted)
            st.write("**LLM Response:**")
            st.code(llm_resp)
