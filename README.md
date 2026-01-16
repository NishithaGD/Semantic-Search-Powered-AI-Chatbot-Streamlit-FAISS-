# Semantic-Search-Powered-AI-Chatbot-Streamlit-FAISS-
# Direct Answer AI Chatbot using Domain Classification & Semantic Search

An AI-powered chatbot that delivers **direct, accurate answers** by first identifying
the domain of a user’s question and then performing **semantic search** within a
domain-specific knowledge base.

Built using **Streamlit**, **FAISS**, and **Transformer models**, this chatbot is designed
for fast, explainable, and controlled responses from curated data.

---

## 🚀 Key Features

- Zero-shot **domain classification** using Transformers
- **Semantic similarity search** using FAISS
- Domain-wise knowledge base indexing
- Pre-written, reliable responses (no hallucinations)
- Clean chat UI using Streamlit
- Scalable architecture for multiple domains

---

## 🧠 How It Works

1. **User enters a question**
2. The chatbot uses a **zero-shot classifier** to detect the most relevant domain
3. Only that domain’s data is searched using **FAISS vector similarity**
4. The most semantically similar question is retrieved
5. The corresponding **pre-written response** is returned instantly

This ensures:
- High accuracy
- Fast responses
- Domain-controlled answers

---

## 🗂 Knowledge Base Structure

The chatbot uses a CSV file named `domain_chatbot.csv` with the following columns:

| Column Name | Description |
|------------|------------|
| `query` | Example or expected user questions |
| `response` | Pre-written answer |
| `domain` | Category of the question |

Each domain gets its **own FAISS index** for optimized search.

---

## 🛠 Technologies Used

- **Python**
- **Streamlit** – Web interface
- **Sentence Transformers** – Text embeddings
- **FAISS** – Fast similarity search
- **Hugging Face Transformers** – Zero-shot classification
- **Pandas & NumPy** – Data handling

---

## ▶️ How to Run
pip install streamlit pandas numpy faiss-cpu sentence-transformers transformers
streamlit run app.py
