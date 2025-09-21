# Conversation Analyst

An AI-powered application that processes earnings call transcript PDFs to extract key insights, generate topics, create summaries, and provide intelligent Q&A using Retrieval-Augmented Generation (RAG).

## Features

* **PDF Extraction & Processing**: Handles extracting raw text from PDF files, identifies speakers, and separates the transcript into "Opening Remarks" and "Q&A" sections.
* **Metadata and Topic Extraction**: Extracts structured data like company name and call date, and generates a list of relevant topics for summarization.
* **Summarization**: Produces concise, business-focused summaries from the extracted content and generated topics.
* **Intelligent Chatbot**: A RAG-based chatbot that uses embeddings from a HuggingFace model and a FAISS vector store to answer user questions about the transcript.

## Technical Overview

The application is built with **Streamlit** for the web interface and uses a modular structure to handle different components:

* **PDF Extraction & Processing**: The `doc_parser.py` utility handles extracting raw text from PDF files, identifying speakers, and separating the transcript into "Opening Remarks" and "Q&A" sections. It removes extraneous information such as headers and footers to ensure the data is clean and ready for analysis.
* **Metadata and Topic Extraction**: The `metadata.py` and `create_topics.py` modules use language models (LLMs) to extract structured data like the company name, call date, and participant information. They also generate a list of relevant topics for summarization.
* **Summarization**: The `create_summary.py` script takes the extracted content and generated topics to produce concise, business-focused summaries.
* **Intelligent Chatbot**: A RAG-based chatbot, implemented in `rag.py`, uses embeddings from a HuggingFace model and a FAISS vector store to answer user questions about the transcript.

## Folder Structure

The repository is organized to separate the main application, LLM calls, and utility functions, making the codebase clean and modular.



### Explanation of Folders and Files

* **`app.py`**: This is the main application file. It uses the Streamlit library to create the web interface and orchestrates the calls to the utility and API functions.
* **`requirements.txt`**: This file lists all the Python dependencies required to run the application. `pip install -r requirements.txt` is used to install them.
* **`README.md`**: The main documentation file for the project, providing an overview, setup instructions, and usage guide.
* **`api/`**: This directory contains Python modules for handling API-related tasks.
    * **`embedding.py`**: Handles the embedding process, likely using the HuggingFace model mentioned in the technical overview to convert text into numerical vectors.
    * **`llm.py`**: Contains code for interacting with Large Language Models (LLMs) to perform tasks like summarization and topic generation.
* **`data/`**: This folder is intended for storing sample PDF transcripts.
* **`faiss_index/`**: This directory stores the FAISS vector store, which is used by the RAG (Retrieval-Augmented Generation) system.
    * **`index.faiss`**: The FAISS index file, which is a highly optimized data structure for efficient similarity search.
    * **`index.pkl`**: A pickle file that likely stores metadata related to the FAISS index, such as the mapping between vectors and the original text chunks.
* **`utils/`**: This directory contains helper scripts and utility functions.
    * **`doc_parser.py`**: The module responsible for parsing the content of the PDF transcripts, cleaning the text, and structuring the data.
    * **`metadata.py`**: Extracts metadata from the documents, such as company name, date, and participants.
    * **`create_topics.py`**: Generates a list of relevant topics from the transcript text.
    * **`create_summary.py`**: Creates a concise summary of the transcript.
    * **`rag.py`**: Implements the RAG system, which uses the FAISS index to retrieve relevant document chunks to answer user questions.

## Setup and Installation

Follow these steps precisely to set up and run the application.

1.  **Clone the Repository**
    First, open your terminal or command prompt and clone the project from GitHub. This command creates a local copy of the repository on your machine.
    ```bash
    git clone [https://github.com/kkarthik3/conversation-analyst.git](https://github.com/kkarthik3/conversation-analyst.git)
    ```

2.  **Navigate to the Project Directory**
    After cloning, you need to change your current directory to the newly created `conversation-analyst` folder. All subsequent commands will be run from this **root directory**.
    ```bash
    cd conversation-analyst
    ```

3.  **Create and Activate a Virtual Environment**
    It's a best practice to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects on your system.
    * **Create the environment**:
        ```bash
        python -m venv venv
        ```
    * **Activate the environment**:
        * On **Windows**:
            ```bash
            venv\Scripts\activate
            ```
        * On **macOS** and **Linux**:
            ```bash
            source venv/bin/activate
            ```
    You will see `(venv)` at the beginning of your terminal prompt, indicating that the virtual environment is active.

4.  **Install Dependencies**
    With the virtual environment active, install all the required libraries listed in `requirements.txt`.
    * **From the root directory (`conversation-analyst`)**, run:
        ```bash
        pip install -r requirements.txt
        ```

## Environment Variable Configuration

The application requires an API token to function.

1.  **Create the `.env` file**:
    * **From the root directory (`conversation-analyst`)**, create a new file and name it `.env`.

2.  **Add the API Token**:
    * Open the `.env` file and add your Hugging Face API token in the specified format:
        ```bash
        HUGGINGFACE="YOUR_HUGGINGFACE_API_TOKEN"
        GROQ_API_KEY="YOUR_GROQ_API_TOKEN"
        ```

## How to Run the Application

Once all the setup is complete, you can start the application.

* **From the root directory (`conversation-analyst`)**, run the following command to launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```
This will open the web application in your default browser.
