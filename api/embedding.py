from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os 
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(model= "sentence-transformers/all-mpnet-base-v2",
                                           huggingfacehub_api_token=os.getenv("HUGGINGFACE"))

def get_embedding(text):
    return embeddings.embed_query(text)


def create_store(documents):
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")

