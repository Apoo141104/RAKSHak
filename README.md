# 🛡️ Rakshak: PII Redaction Streamlit Application

### Secure AI Interaction for Sensitive Data

---

## 📈 **Project Overview**

**Rakshak** is a multi-layered **Streamlit application** designed to **detect, redact, and assess** Personally Identifiable Information (PII) in user text. It ensures that **only anonymized content** is sent to Large Language Models (LLMs), enabling AI-driven interactions **without compromising user privacy**.

---

## ✨ **Features**

* 🔍 **Multi-Layered Redaction**
  Combines regex-based detection and advanced Named Entity Recognition (NER).

* 🇮🇳 **Indian-Specific PII Patterns**
  Detects Aadhaar, PAN, Voter ID, Passport, Vehicle Reg No., Employee IDs.

* 🧐 **Contextual NER via HuggingFace**
  Uses [`ai4bharat/IndicNER`](https://huggingface.co/ai4bharat/IndicNER) for contextual PII detection in Indian languages.

* ⚠️ **Sensitivity Detection**
  Uses `facebook/bart-large-mnli` to classify text as `confidential`, `personal`, or `public`.

* 🤖 **Safe LLM Integration**
  Only the redacted version of the input is sent to Groq’s Llama3 for processing.

* 🖥️ **Interactive UI with Streamlit**
  Simple web app showing PII entities, redacted text, and LLM responses.

---

## 🛠️ **How It Works: Layered Protection**

### 1️⃣ **Presidio Analyzer (Pattern Detector)**

* **Detects**: Email, phone numbers, Aadhaar, PAN, etc.
* **Effect**: Replaces values with tags like `[EMAIL_ADDRESS_1]`.

### 2️⃣ **HuggingFace NER (Contextual Understander)**

* **Detects**: Names, locations, organizations from text context.
* **Effect**: `John Doe` → `[PERSON_1]`.

### 3️⃣ **Sensitivity Classifier (Risk Assessment)**

* **Classifies**: Entire input text as `confidential`, `personal`, or `public`.
* **Effect**: Doesn't redact — provides a sensitivity label.

### 4️⃣ **LLM Integration (Secure Querying)**

* **LLM Used**: Groq’s Llama3.
* **Effect**: Receives only the fully redacted text — not original content.

---

## 🚀 **Local Setup & Run**

Follow these steps to run the app on your machine.

### 🔧 **Prerequisites**

* Python 3.8 or newer (Tested with Python 3.11)
* Git installed

### 📅 Step 1: Clone the Repository

```bash
git clone https://github.com/Apoo141104/RAKSHak.git
cd RAKSHak
```

### 🧱 Step 2: Create and Activate a Virtual Environment

#### Create a virtual environment:

```bash
python -m venv venv
```

#### Activate it:

* macOS / Linux:

```bash
source venv/bin/activate
```

* Windows (Command Prompt):

```cmd
.\venv\Scripts\activate
```

* Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

> ✅ You’ll see `(venv)` in your terminal prompt when activated.

### 📦 Step 3: Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

#### Sample `requirements.txt` content:

```txt
streamlit
presidio-analyzer
python-dotenv
transformers
torch
groq
sentencepiece
accelerate
```

### 🔑 Step 4: Set Up Groq API Key

To connect with the LLM securely:

#### Create a `.streamlit` folder:

```bash
mkdir .streamlit
```

#### Inside it, create `secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

> 🔐 Replace with your real API key from [Groq Console](https://console.groq.com).

### ▶️ Step 5: Run the Streamlit App

```bash
streamlit run app.py
```

> 🌐 Opens in your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Testing Your Application

### 🔹 Comprehensive Input

```txt
Urgent internal memo: This document contains highly confidential information. Patient Anjali Sharma (DOB: 15/03/1988) visited Apollo Hospital on 2024-06-20 for follow-up. Her unique patient ID is PX7890123. The physician Dr. Rajesh Kumar (Mobile: +919876543210) noted her AADHAAR number 9876 5432 1098. She works for TechSolutions India. Employee ID E12ABU5678 is assigned to Mr. Vikram Singh, a senior analyst at State Bank of India. His PAN is ABCDE1234F and email is vikram.singh@examplebank.com. We also received a query from a Ministry of Defense official regarding vehicle registration DL01CD1234 for a new project located near the Air Force Station, Hindon. Please ensure all PII and sensitive project details are redacted before sharing any summaries. Contact our legal department at legal@techsolutions.com for further clarification. Voting ID of Ms. Priya Patel: ABC1234567. Passport No. K1234567.
```

### 🔹 Short Input: Personal Info

```txt
Could you please help me confirm my identity? My full name is Sarah Miller, and my private phone number is +1-202-555-0100. Thanks.
```

### 🔹 Short Input: Government Sensitive Info

```txt
Please redact details from this secure document. The official government ID for the operation is GVT-SEC-98765, linked to agent E78XYZ4321.
```

---

## 🤝 Contributing

* Fork the repository
* Create a new branch
* Commit changes
* Submit a pull request

> For significant changes, open an issue to start a discussion.

---


