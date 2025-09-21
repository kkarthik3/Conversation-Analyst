from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
import os
import streamlit as st

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

llm_oss = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
) 

groq_chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
)

chat_oss = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0
) 