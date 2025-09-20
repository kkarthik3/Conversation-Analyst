from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
import os
import streamlit as st

from langchain_groq import ChatGroq

llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}},
    api_key="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")