# Azerbaijani AI Strategy FAQ RAG System

A complete Retrieval-Augmented Generation (RAG) system for answering questions about Azerbaijan's 2025-2028 Artificial Intelligence Strategy. The system processes PDF documents containing FAQs in Azerbaijani language and provides semantically relevant answers enhanced with LLM generation.

## Features

- **PDF Processing**: Automatically extracts FAQs from PDF documents
- **Semantic Search**: Uses LaBSE (Language-agnostic BERT Sentence Embedding) to find relevant answers
- **LLM Generation**: Enhances retrieved answers with local language models via Ollama
- **Azerbaijani Language Support**: Handles Azerbaijani language specifics including character variations
- **User-Friendly Interface**: Interactive Streamlit web application
- **Dockerized Deployment**: Complete containerization with Docker Compose

## Project Structure

```
az-faq-rag/
│
├── app.py                    # Main Streamlit application
├── az_faq_rag.py             # Retrieval module (embedding-based search)
├── ollama_helper.py          # Helper functions for Ollama LLM integration
├── pdf_extractor.py          # PDF processing and FAQ extraction
│
├── data/                     # Data directory
│   └── azerbaijani_ai_faqs.csv  # Extracted FAQs in CSV format
│
├── models/                   # Cached models directory
│
├── uploads/                  # Directory for uploaded PDF files
│
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
├── requirements.txt          # Python dependencies
│
└── README.md                 # Project documentation
```

## How It Works

The system follows a standard RAG workflow with these key components:

1. **PDF Processing**:
   - Uploads PDF documents containing Azerbaijani FAQs
   - Extracts questions and answers using heuristic patterns
   - Saves the extracted FAQs to a CSV file for efficient processing

2. **Retrieval**:
   - Converts user queries and FAQs to vector embeddings using LaBSE
   - Finds the most semantically similar FAQs to the user's question
   - Ranks results by similarity score

3. **Generation**:
   - Passes the retrieved relevant FAQs as context to a local LLM
   - Generates a comprehensive response based on the context
   - Ensures responses are appropriate and based only on the provided information

4. **Azerbaijani Language Support**:
   - Handles special characters and different writing styles for Azerbaijani
   - Processes 'ş', 'ç', 'ə', 'ö', 'ü' and alternative transliterations
   - Optimized for semantic searching in Azerbaijani language

## Why CSV Files?

The system uses CSV files (`data/azerbaijani_ai_faqs.csv`) as an intermediate storage format for extracted FAQs because:

1. **Efficiency**: Parsing PDFs is computationally expensive, so we extract FAQs once and store them in a structured format
2. **Indexing**: CSV allows for quick loading and indexing of questions and answers
3. **Persistence**: Saves the state between application restarts
4. **Structured Data**: Maintains the relationship between questions, answers, and categories

The PDF extractor automatically processes PDF documents and converts them to this CSV format for the RAG system.

## Installation

### Using Docker (Recommended)

1. Make sure you have Docker and Docker Compose installed
2. Clone this repository
3. Build and run the containers:

```bash
docker-compose up -d
```

This will start two containers:
- The RAG application on port 8501
- Ollama LLM service on port 11434

### Manual Installation

1. Install Python 3.9 or newer
2. Install Ollama from [ollama.com](https://ollama.com/)
3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

## Setup and Usage

1. **Start the application**: 
   - Docker: `docker-compose up -d`
   - Manual: `streamlit run app.py`

2. **Access the web interface**:
   - Open your browser and go to `http://localhost:8501`

3. **Upload a PDF document**:
   - Use the sidebar PDF uploader
   - Click "Process PDF" to extract FAQs

4. **Download an LLM model** (if using Ollama):
   - Enter model name in the sidebar (e.g., `mistral:latest`)
   - Click "Download Model"
   - Wait for the download to complete

5. **Search for information**:
   - Enter your question in Azerbaijani in the search field
   - Choose the number of results to retrieve
   - Select whether to generate an LLM response
   - Click "Search"

6. **View and add FAQs**:
   - Use the "All FAQs" tab to browse existing questions and answers
   - Use the "Add New FAQ" tab to manually add new entries

## LLM Models

The system works with any Ollama-compatible LLM. Recommended models include:

- **mistral:latest** - Good performance with modest resource requirements
- **llama2:7b** - Balanced performance and resource usage
- **phi:latest** - Lightweight option for systems with limited resources

## System Requirements

### Minimum Requirements
- 4GB RAM
- 10GB free disk space
- Dual-core CPU

### Recommended Requirements
- 16GB RAM
- 20GB free disk space
- Quad-core CPU
- NVIDIA GPU with 4GB+ VRAM (for faster LLM inference)

## Customization

### Changing the System Prompt

Modify the system prompt in the sidebar to customize how the LLM responds. For Azerbaijani-specific responses, include instructions in Azerbaijani language.

### Adjusting PDF Extraction

The `pdf_extractor.py` file contains the logic for extracting FAQs from PDFs. You can customize the extraction patterns to better match your specific PDF format.

### Docker Configuration

Edit the `docker-compose.yml` file to:
- Change exposed ports
- Enable GPU support (uncomment the GPU section)
- Configure resource limits
- Mount additional volumes

## Troubleshooting

### Common Issues

1. **PDF extraction not finding questions**:
   - Check that your PDF follows a clear question-answer format
   - Edit the question patterns in `pdf_extractor.py`

2. **Ollama connection issues**:
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is downloaded: `ollama list`

3. **Out of memory errors**:
   - Try a smaller LLM model
   - Reduce the batch size in the app configuration
   - Increase your system's swap space

### Docker Issues

1. **Container not starting**:
   - Check logs: `docker-compose logs`
   - Ensure ports are not already in use

2. **GPU not detected**:
   - Verify NVIDIA drivers are installed
   - Ensure nvidia-docker is properly configured

## License

This project is released under the MIT License.

## Acknowledgments

- LaBSE model by Google Research
- Ollama by Ollama.ai
- Streamlit for the web interface
- PyPDF2 for PDF processing
