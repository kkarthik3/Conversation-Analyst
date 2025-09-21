from api.llm import *

from typing import List
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser


# Define input schema with Pydantic (optional, but helps with validation)

# Create a prompt template
prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant. Summarize the following text 
    **without changing its meaning or context**.

    Focus only on the following topics:
    {topics}

    Text to summarize:
    {text}

    Provide a structured summary organized by the given topics.
    """
)

# Define an output parser (parses LLM response into a string)
parser = StrOutputParser()

# Define summarizer as a chain
def summarizer(input:str,topics:str):
    extractorchain = prompt | groq_chat | StrOutputParser() 
    result = extractorchain.invoke({"text":input,"topics":topics})
    return result




