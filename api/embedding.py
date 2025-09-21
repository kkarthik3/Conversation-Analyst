"""FAISS VECTOR STORE MANAGEMENT.

Create and manage a FAISS vector store 
using Hugging Face embeddings. It provides functions to generate text 
embeddings and save documents to a local FAISS index.
"""

from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os 
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from dotenv import load_dotenv

# Load environment variables from a .env file.
load_dotenv()

# Initialize the HuggingFaceEndpointEmbeddings model.
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE")
)

def get_embedding(text: str) -> list[float]:
    """Generates a vector embedding for a given text query.

    Args:
        text: The input text string to be embedded.

    Returns:
        A list of floats representing the vector embedding of the text.
    """
    # The 'embed_query' method is used to get the embedding for a single text.
    return embeddings.embed_query(text)

def create_store(documents: list[Document]):
    """Creates a FAISS vector store from a list of documents and saves it locally.

    This function takes a list of LangChain Document objects, generates
    embeddings for each, and builds an in-memory FAISS index. The index is
    then persisted to disk for later use.

    Args:
        documents: A list of Document objects to be added to the vector store.
    """
    # FAISS.from_documents is a convenient method that handles the embedding
    # of documents and the creation of the FAISS index in a single step.
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save the vector store to a local directory named "faiss_index".
    # This allows for the index to be loaded later without re-processing documents.
    vector_store.save_local("faiss_index")