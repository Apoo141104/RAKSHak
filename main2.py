import os
import torch
import streamlit as st
from dotenv import load_dotenv
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Setup Presidio for custom + built-in recognizers ---
analyzer = AnalyzerEngine()

# --- Add custom Indian patterns ---
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

# --- Load HuggingFace NER model (IndicNER) ---
model_name = "ai4bharat/IndicNER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

ner_pipe = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0 if device == "cuda" else -1
)

# --- Sensitivity classification using zero-shot ---
try:
    sensitivity_model = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if device == "cuda" else -1
    )
except Exception as e:
    st.error(f"Failed to load sensitivity model: {str(e)}")
    sensitivity_model = None

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

# --- Redaction using NER ---
NER_TAGS = {
    "PER": "PERSON",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "MISC": "MISC"
}

def redact_ner(text):
    results = ner_pipe(text)
    redacted = text
    entity_counts = {}
    for ent in sorted(results, key=lambda x: x['start'], reverse=True):
        entity_type = NER_TAGS.get(ent['entity_group'], ent['entity_group'])
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        tag = f"[{entity_type}_{entity_counts[entity_type]}]"
        redacted = redacted[:ent['start']] + tag + redacted[ent['end']:]
    return redacted, results

# --- Redaction using Presidio ---
def redact_presidio(text):
    entities_to_detect = list(custom_patterns.keys()) + ["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON"]
    results = analyzer.analyze(text=text, entities=entities_to_detect, language="en")
    redacted = text
    entity_counts = {}
    for r in sorted(results, key=lambda x: x.start, reverse=True):
        entity_type = r.entity_type
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        tag = f"[{entity_type}_{entity_counts[entity_type]}]"
        redacted = redacted[:r.start] + tag + redacted[r.end:]
    return redacted, results

# --- Streamlit UI ---
st.set_page_config(page_title="PII Redactor", layout="wide")
st.title("🛡️ Unified PII Protection: Regex + NER + Sensitivity Detection")

# Display default Presidio recognizers
st.subheader("📦 Default Presidio Recognizers")
built_in_entities = [rec.supported_entities for rec in analyzer.registry.recognizers if "IN_" not in str(rec.supported_entities)]
flattened = [item for sublist in built_in_entities for item in sublist]
st.code(", ".join(sorted(set(flattened))))

# User input
user_input = st.text_area("✏️ Enter your text here:", height=200)

if st.button("🔐 Analyze & Redact"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Apply Presidio regex-based redaction
        presidio_redacted, presidio_entities = redact_presidio(user_input)

        # Apply HuggingFace NER-based redaction
        ner_redacted, ner_entities = redact_ner(presidio_redacted)

        # Display Presidio results
        st.subheader("🔍 Detected Presidio Entities")
        if presidio_entities:
            for entity in presidio_entities:
                st.write(f"- {entity.entity_type}: {user_input[entity.start:entity.end]} (pos: {entity.start}-{entity.end})")
        else:
            st.write("✅ No entities detected by Presidio")

        # Display HuggingFace NER results
        st.subheader("🔍 Detected HuggingFace NER Entities")
        if ner_entities:
            for ent in ner_entities:
                entity_type = NER_TAGS.get(ent['entity_group'], ent['entity_group'])
                st.write(f"- {entity_type}: {ent['word']} (pos: {ent['start']}-{ent['end']})")
        else:
            st.write("✅ No entities detected by HuggingFace NER")

        # Sensitivity detection
        if is_sensitive(user_input):
            st.warning("⚠️ Model flagged this content as sensitive or confidential.")

        # Final redacted output
        st.subheader("🧼 Final Redacted Text")
        st.code(ner_redacted)

        # Placeholder for Groq API
        st.subheader("🤖 LLM (Groq) Response")
        st.info("🔧 Placeholder: Add your Groq API integration here.")
