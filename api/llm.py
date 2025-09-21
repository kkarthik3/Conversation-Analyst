
"""LLM CHAINING AND STREAMLIT APPLICATION.

This script sets up a basic LangChain application using two different Groq-powered
language models and will be used internally for summarization, topic modelling and etc..
"""


from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
import os
import streamlit as st

from langchain_groq import ChatGroq

from dotenv import load_dotenv

# Load environment variables from a .env file.
load_dotenv()

# --- Language Model Configuration ---

# ChatGroq instance configured for JSON output.
llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Another ChatGroq instance, this one using the 'openai/gpt-oss-120b' model.
llm_oss = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# ChatGroq instance for general conversational use (not JSON).
# Uses the 'llama-3.3-70b-versatile' model for standard chat responses.
groq_chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
)

# This one is based on the 'openai/gpt-oss-120b' model.
chat_oss = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0
)

