from langchain_community.document_loaders import PyMuPDFLoader
import os
import streamlit as st

# Get the current working directory
current_dir = os.getcwd()

# Build relative path (assuming the PDF is in the same folder as the script)
file_path = os.path.join(current_dir, "Q2FY24_LaurusLabs_EarningsCallTranscript.pdf")


@st.cache_data()
def file_load(path:str, default:bool):
    if default :
        loader = PyMuPDFLoader(file_path)
    else:
        loader = PyMuPDFLoader(path)

    return loader.load()

