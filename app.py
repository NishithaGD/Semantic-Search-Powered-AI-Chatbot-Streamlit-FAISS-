import streamlit as st
import pandas as pd
import numpy as np
import faiss #facebook AI similarity search
from sentence_transformers import SentenceTransformer #converts sentence into vectors
from transformers import pipeline

# Set the page configuration
st.set_page_config(
    page_title="Direct Answer AI Chatbot",
    page_icon="✅",
    layout="wide"
)

# --- MODEL LOADING (with caching for performance) ---

@st.cache_resource
def load_models():
    """
    Loads the necessary models: an embedding model for search and a classifier for domain detection.
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # This classifier will categorize the user's query into one of our domains
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return embedding_model, classifier

embedding_model, classifier = load_models()

# --- KNOWLEDGE BASE INITIALIZATION ---

@st.cache_resource
def initialize_knowledge_base():
    """
    Loads data from the CSV and builds a separate FAISS index for each domain's 'query' column.
    This allows for fast and filtered semantic searching.
    """
    try:
        df = pd.read_csv('domain_chatbot.csv', encoding='latin-1')
        # We now use 'query' and 'response' as per your CSV file.
        required_columns = ['query', 'response', 'domain']
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain the following columns: {required_columns}")
            return None, None

        df.dropna(subset=required_columns, inplace=True)
        
        domain_indices = {}
        unique_domains = df['domain'].unique()

        for domain in unique_domains:
            with st.spinner(f"Building knowledge base for '{domain}' domain..."):
                domain_df = df[df['domain'] == domain].copy().reset_index(drop=True)
                
                # Embed only the 'query' column for searching
                queries = domain_df['query'].tolist()
                corpus_embeddings = embedding_model.encode(queries, convert_to_tensor=True)
                
                index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
                index.add(corpus_embeddings.cpu().numpy())
                
                domain_indices[domain] = {
                    'index': index,
                    'dataframe': domain_df # Store the dataframe to retrieve the answer later
                }
        
        return domain_indices, unique_domains.tolist()

    except FileNotFoundError:
        st.error("`domain_chatbot.csv` not found. Please make sure it's in the same folder.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during initialization: {e}")
        return None, None

domain_knowledge, domain_labels = initialize_knowledge_base()

# --- CORE CHATBOT FUNCTIONS ---

def get_answer(user_query, labels):
    """
    The main function to get the final answer.
    1. Classifies the domain of the query.
    2. Searches the corresponding knowledge base for the most similar question.
    3. Returns the pre-written answer for that question.
    """
    # 1. Classify the domain
    result = classifier(user_query, candidate_labels=labels)
    predicted_domain = result['labels'][0]

    # 2. Search within the classified domain
    if predicted_domain in domain_knowledge:
        domain_data = domain_knowledge[predicted_domain]
        index = domain_data['index']
        df = domain_data['dataframe']
        
        # Find the most similar question in the knowledge base
        query_embedding = embedding_model.encode([user_query], convert_to_tensor=True)
        distances, indices = index.search(query_embedding.cpu().numpy(), k=1) # Search for the single best match
        
        # 3. Retrieve the corresponding answer
        best_match_index = indices[0][0]
        answer = df.iloc[best_match_index]['response']
        
        return answer, predicted_domain
        
    return "I'm sorry, I couldn't find a relevant domain for your question.", "N/A"

# --- STREAMLIT UI ---

st.title("✅ Direct Answer AI Chatbot")
st.markdown("This chatbot identifies the question's category and provides the best pre-written answer directly from its knowledge base.")

if domain_knowledge:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Finding the best answer..."):
                response, domain = get_answer(prompt, domain_labels)
                
                st.markdown(response)
                st.info(f"Detected Domain: `{domain}`", icon="🏷️")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("Chatbot initialization failed. Please check the error messages above.")