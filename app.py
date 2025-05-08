# Complete RAG Streamlit application with LLM integration and PDF support

import streamlit as st
import pandas as pd
import os
import re
from az_faq_rag import AzerbaijaniFAQRAG
import io
from pdf_extractor import PDFFAQExtractor
from ollama_helper import (
    check_ollama_installed, 
    check_ollama_running, 
    start_ollama_service,
    check_model_exists,
    pull_model,
    generate_response
)

# Page config
st.set_page_config(
    page_title="Azerbaijani AI Strategy FAQ RAG",
    page_icon="🤖",
    layout="wide"
)

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

FAQ_FILE = 'data/azerbaijani_ai_faqs.csv'

# Default LLM settings
DEFAULT_MODEL = "mistral:latest"
DEFAULT_SYSTEM_PROMPT = """
You are an expert assistant on Azerbaijan's Artificial Intelligence Strategy.
You provide precise and informative answers to questions in Azerbaijani language.
If you don't know the answer, admit it honestly and don't make things up.
Your answers should be based on the provided context.
"""

# Initialize session state for LLM settings
if 'model_name' not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 1000
if 'ollama_available' not in st.session_state:
    st.session_state.ollama_available = False
if 'model_available' not in st.session_state:
    st.session_state.model_available = False

# Initialize RAG system
@st.cache_resource
def load_rag_system():
    """Load the RAG system with caching for better performance"""
    rag = AzerbaijaniFAQRAG(cache_dir='models')
    
    # Check if FAQ file exists and load it
    if os.path.exists(FAQ_FILE):
        try:
            rag.load_faqs_from_csv(FAQ_FILE)
            st.session_state['faq_source'] = FAQ_FILE
        except Exception as e:
            st.error(f"Error loading FAQs: {str(e)}")
    
    return rag

# Initialize PDF extractor
pdf_extractor = PDFFAQExtractor(output_dir='data')

# Initialize the rag system
rag = load_rag_system()

# Check if Ollama is available and start it if possible
def setup_ollama():
    if check_ollama_installed():
        st.session_state.ollama_available = True
        
        if not check_ollama_running():
            with st.spinner("Starting Ollama service..."):
                if start_ollama_service():
                    st.success("Ollama service started successfully!")
                else:
                    st.error("Failed to start Ollama service. Please start it manually.")
                    st.session_state.ollama_available = False
                    return
        
        # Check if model exists
        if not check_model_exists(st.session_state.model_name):
            st.warning(f"{st.session_state.model_name} model not found. It's recommended to download it.")
            st.session_state.model_available = False
        else:
            st.session_state.model_available = True
    else:
        st.session_state.ollama_available = False
        st.warning("Ollama is not installed. For the full RAG experience, download Ollama from https://ollama.com/")

# Setup Ollama
setup_ollama()

# Function to handle PDF upload and processing
def handle_pdf_upload(uploaded_file):
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        
        # Process the PDF
        with st.spinner("Extracting FAQs from PDF..."):
            try:
                csv_path = pdf_extractor.process_pdf(file_path, output_file=FAQ_FILE)
                st.success(f"FAQs extracted and saved to: {csv_path}")
                
                # Reload the RAG system with new FAQs
                rag.load_faqs_from_csv(csv_path)
                st.session_state['faq_source'] = csv_path
                
                # Display the number of FAQs extracted
                st.info(f"{len(rag.faqs)} FAQs extracted from the PDF")
                
                return True
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                return False
    return False

# Sidebar for LLM settings and PDF upload
with st.sidebar:
    st.title("RAG System Settings")
    
    # PDF Upload Section
    st.header("PDF Upload")
    uploaded_file = st.file_uploader("Upload an Azerbaijani PDF document", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            handle_pdf_upload(uploaded_file)
    
    # LLM Settings Section
    st.header("LLM Settings")
    if st.session_state.ollama_available:
        st.session_state.model_name = st.text_input("Model name:", value=st.session_state.model_name)
        
        if not st.session_state.model_available:
            if st.button("Download Model"):
                with st.spinner(f"Downloading {st.session_state.model_name} model..."):
                    if pull_model(st.session_state.model_name):
                        st.success(f"{st.session_state.model_name} model downloaded successfully!")
                        st.session_state.model_available = True
                    else:
                        st.error(f"Failed to download {st.session_state.model_name} model.")
        
        st.session_state.system_prompt = st.text_area("System prompt:", value=st.session_state.system_prompt, height=200)
        st.session_state.temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.1)
        st.session_state.max_tokens = st.number_input("Max tokens:", value=st.session_state.max_tokens, min_value=100, max_value=4000, step=100)
    else:
        st.info("LLM settings will be available after installing Ollama.")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This system is designed to answer questions about Azerbaijan's 2025-2028 Artificial Intelligence Strategy.")
    st.markdown("The project combines LaBSE embedding model and a local LLM to provide a comprehensive Retrieval-Augmented Generation (RAG) system.")

# App title and header
st.title("🤖 Azerbaijani AI Strategy FAQ RAG")
st.markdown("### Question-Answering System for Azerbaijan's 2025-2028 Artificial Intelligence Strategy")

# Display FAQ source if available
if 'faq_source' in st.session_state:
    st.markdown(f"*Using FAQs from: {st.session_state['faq_source']}*")

# App tabs
tab1, tab2, tab3, tab4 = st.tabs(["RAG Search", "Simple Search", "Add New FAQ", "All FAQs"])

# RAG Search tab
with tab1:
    st.subheader("Enter your question about the AI strategy:")
    
    query = st.text_area("", placeholder="Example: What are the priority directions in the AI strategy?", height=100, label_visibility="collapsed")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        top_k = st.number_input("Number of results:", min_value=1, max_value=10, value=3, step=1)
    
    with col2:
        generate_llm_response = st.checkbox("Generate LLM response", value=True, disabled=not st.session_state.model_available)
    
    with col3:
        search_btn = st.button("Search", type="primary", use_container_width=True)
    
    # Handle search button
    if search_btn and query:
        if rag.faqs is None or len(rag.faqs) == 0:
            st.warning("No FAQs available. Please upload a PDF document or add FAQs manually.")
        else:
            with st.spinner("Searching..."):
                results = rag.retrieve(query, top_k=top_k)
            
            # Display results
            st.subheader(f"Results for '{query}':")
            
            # Display retrieved results
            if results:
                # Prepare context for LLM
                context = ""
                for i, result in enumerate(results):
                    similarity_pct = round(result["similarity"] * 100, 1)
                    
                    with st.expander(f"**{i+1}. {result['question']}** (Similarity: {similarity_pct}%)", expanded=False):
                        st.markdown(f"**Answer:** {result['answer']}")
                        if "category" in result:
                            st.markdown(f"**Category:** {result['category']}")
                    
                    # Add to context for LLM
                    context += f"Question: {result['question']}\nAnswer: {result['answer']}\n\n"
                
                # Generate LLM response if enabled and available
                if generate_llm_response and st.session_state.model_available:
                    st.markdown("---")
                    st.subheader("LLM Response:")
                    
                    with st.spinner("Generating LLM response..."):
                        # Prepare prompt with context
                        prompt = f"""
Using the following information database, provide an accurate, clear, and comprehensive answer to this question:

Question: {query}

Information Database:
{context}

If the question cannot be fully answered with the provided information, use only the context given and do not add information outside of this context. If you don't know the exact answer, acknowledge this.
                        """
                        
                        # Generate response
                        llm_response = generate_response(
                            model_name=st.session_state.model_name,
                            prompt=prompt,
                            system_prompt=st.session_state.system_prompt,
                            temperature=st.session_state.temperature,
                            max_tokens=st.session_state.max_tokens
                        )
                        
                        st.markdown(llm_response)
            else:
                st.warning("No results found.")

# Simple Search tab
with tab2:
    st.subheader("Simple search for AI strategy questions:")
    
    query_simple = st.text_input("", placeholder="Example: How will AI be applied in Azerbaijan?", label_visibility="collapsed")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        top_k_simple = st.number_input("Number of results:", min_value=1, max_value=10, value=3, step=1, key="top_k_simple")
    
    with col2:
        search_btn_simple = st.button("Search", type="primary", use_container_width=True, key="search_btn_simple")
    
    # Handle search button
    if search_btn_simple and query_simple:
        if rag.faqs is None or len(rag.faqs) == 0:
            st.warning("No FAQs available. Please upload a PDF document or add FAQs manually.")
        else:
            with st.spinner("Searching..."):
                results = rag.retrieve(query_simple, top_k=top_k_simple)
            
            # Display results
            st.subheader(f"Results for '{query_simple}':")
            
            if results:
                for i, result in enumerate(results):
                    similarity_pct = round(result["similarity"] * 100, 1)
                    
                    with st.expander(f"**{i+1}. {result['question']}** (Similarity: {similarity_pct}%)", expanded=True):
                        st.markdown(f"**Answer:** {result['answer']}")
                        if "category" in result:
                            st.markdown(f"**Category:** {result['category']}")
            else:
                st.warning("No results found.")

# Add new FAQ tab
with tab3:
    st.subheader("Add new question and answer")
    
    new_question = st.text_area("Question:", height=100)
    new_answer = st.text_area("Answer:", height=200)
    new_category = st.text_input("Category:", value="Artificial Intelligence")
    
    add_btn = st.button("Add", type="primary")
    
    if add_btn:
        if new_question and new_answer:
            if rag.faqs is None:
                # Initialize FAQs if none exist
                rag.add_faq(new_question, new_answer, new_category)
            else:
                rag.add_faq(new_question, new_answer, new_category)
            
            rag.save_faqs(FAQ_FILE)
            st.success("FAQ added successfully!")
            
            # Clear inputs
            st.session_state['new_question'] = ""
            st.session_state['new_answer'] = ""
        else:
            st.error("Question and answer must be provided.")

# View all FAQs tab
with tab4:
    st.subheader("All questions and answers")
    
    if rag.faqs is not None and len(rag.faqs) > 0:
        # Group by category
        if 'category' in rag.faqs.columns:
            categories = rag.faqs['category'].unique()
            
            for category in categories:
                with st.expander(f"**{category}**", expanded=True):
                    category_faqs = rag.faqs[rag.faqs['category'] == category]
                    
                    for i, (_, row) in enumerate(category_faqs.iterrows()):
                        st.markdown(f"**Question {i+1}:** {row['question']}")
                        st.markdown(f"**Answer:** {row['answer']}")
                        st.markdown("---")
        else:
            # If no categories, just show all FAQs
            for i, (_, row) in enumerate(rag.faqs.iterrows()):
                st.markdown(f"**Question {i+1}:** {row['question']}")
                st.markdown(f"**Answer:** {row['answer']}")
                st.markdown("---")
    else:
        st.warning("No questions available. Please upload a PDF document or add FAQs manually.")

# Footer
st.markdown("---")
st.markdown("#### Complete RAG System using LaBSE local embedding model and Ollama LLM")
st.markdown("*Note: This system is designed based on Azerbaijan's 2025-2028 Artificial Intelligence Strategy.*")
