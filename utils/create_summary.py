"""TEXT SUMMARIZATION CHAIN.

This module defines a LangChain-based text summarization pipeline.
It uses a prompt template to guide a language model to summarize text
based on specific, user-defined topics. The chain is encapsulated
in a reusable function.
"""

from api.llm import *
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser



# Create a prompt template for the summarization task.
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

# Define an output parser.
# StrOutputParser is a simple parser that converts the LLM's output
parser = StrOutputParser()

# Define the summarizer as a chain.
def summarizer(input: str, topics: str) -> str:
    """Summarizes a given text focusing on specific topics.

    This function orchestrates a LangChain expression language (LCEL) chain
    that first formats the prompt with the input text and topics,
    then passes it to the 'groq_chat' language model, and finally
    parses the output into a string.

    Args:
        input: The text to be summarized.
        topics: A string or list of strings representing the topics
                to focus on during summarization.

    Returns:
        The summarized text as a string, structured by the specified topics.
    """
    # Create the chain: prompt | LLM | output parser.
    # The 'invoke' method executes the chain with the provided inputs.
    extractorchain = prompt | groq_chat | StrOutputParser()
    
    # Invoke the chain with the text and topics.
    result = extractorchain.invoke({"text": input, "topics": topics})
    
    # Return the final summarized string.
    return result