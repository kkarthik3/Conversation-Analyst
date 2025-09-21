"""TOPIC EXTRACTION CHAIN.

This module defines a LangChain pipeline for extracting key topics from text.
It uses a Pydantic model for schema validation, a prompt template to guide
the language model, and a custom safe parsing function to handle potential
JSON formatting errors from the LLM. The final output is a JSON object
containing a list of extracted topics.
"""

from typing import List
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import json

from api.llm import llm_oss, llm_groq # your initialized LLM instance

# --- Pydantic Model Definition ---

# Define Pydantic model for the output structure.
class TopicsOutput(BaseModel):
    """Defines the expected JSON output schema for topic extraction.

    This model enforces that the output is a JSON object with a single key
    "topics" which contains a list of strings. This provides a robust
    schema for validation.
    """
    topics: List[str] = Field(..., description="List of relevant topics extracted from context")


# --- Prompt Template ---

# Prompt template for the topic extraction task.
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are an assistant that extracts Top 5 relevant key topics from the given context which is relevant.
    
    Context: {text}
    
    Provide a JSON object with a single key "topics" and a list of relevant topics.
    
    Example output:
    {{
        "topics": [
            "Artificial intelligence",
            "Healthcare",
            "Finance",
            "Transportation"
        ]
    }}
    """
)


# --- Error Handling and Parsing ---

# Default dictionary in case parsing fails.
# This ensures that the function always returns a predictable structure,
# preventing downstream errors in the application.
default_dict = {
    "topics": []
}

#  Safe parse function.
def safe_parse(response: str, model: BaseModel, default: dict) -> dict:
    """Safely parses a JSON string into a Pydantic model instance.

    This function attempts to validate the response string against the
    Pydantic model. If validation or JSON decoding fails, it returns a
    pre-defined default dictionary.

    Args:
        response: The JSON string to parse.
        model: The Pydantic model to validate against.
        default: The default dictionary to return on failure.

    Returns:
        A dictionary representing the parsed data or the default dictionary.
    """
    try:
        # Pydantic's model_validate_json method is used for validation.
        return model.model_validate_json(response)
    except json.JSONDecodeError:
        # Handles cases where the LLM's output is not valid JSON.
        return default
    except Exception:
        # Catches other potential validation errors from Pydantic.
        return default


# --- LangChain Chain Definition ---

# Build the extraction chain using LangChain Expression Language (LCEL).
# The chain consists of three main components:
# 1. prompt: Formats the input text into the prompt template.
# 2. llm_oss: Passes the formatted prompt to the LLM for a JSON response.
# 3. StrOutputParser(): Converts the LLM response object into a simple string.
# 4. RunnableLambda: Applies the 'safe_parse' function to the string output.
#    This is a crucial step for robust error handling.
extractorchain = prompt | llm_oss | StrOutputParser() | RunnableLambda(
    lambda r: safe_parse(r, TopicsOutput, default_dict)
)


# --- Main Function ---

#  Function to get topics from documents.
def extract_topics(docs: List) -> dict:
    """Extracts key topics from a list of document objects.

    This function combines the content of all documents into a single string,
    then invokes the pre-defined LangChain pipeline to extract relevant
    topics in a structured format.

    Args:
        docs: A list of document objects, where each object must have a
              `.page_content` attribute containing the text.

    Returns:
        A dictionary containing the extracted topics, with the format
        {"topics": ["topic1", "topic2", ...]}.
    """
    # Combine the content of all documents into a single large string.
    combined_text = " ".join([doc.page_content for doc in docs])
    
    # Invoke the LangChain extractor chain with the combined text.
    result = extractorchain.invoke({
        "text": combined_text
    })

    # Convert the Pydantic model instance `result` to a JSON string and then
    # back into a standard Python dictionary to ensure the final output
    # is a simple dictionary, as specified by the function's return type.
    result_json = result.model_dump_json(indent=2)
    return json.loads(result_json)