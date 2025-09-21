"""METADATA EXTRACTION CHAIN.

This module defines a LangChain pipeline for extracting key metadata from
conference call transcripts, such as company name, call date, and management
participants. It leverages a Pydantic model for robust schema enforcement
and includes a custom, safe parsing function to handle potential LLM
output formatting errors.
"""

from typing import List
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from datetime import date, datetime
import json

from api.llm import llm_groq

# --- Pydantic Model Definitions ---

# Defines the schema for a single conference call participant.
class Participant(BaseModel):
    """Represents a single management participant with name and designation."""
    name: str = Field(..., description="Full name of the management participant")
    designation: str = Field(..., description="Designation of the participant")

# Defines the top-level schema for conference call metadata.
class ConferenceCall(BaseModel):
    """Schema for extracting company, date, and participant information."""
    company_name: str = Field(..., description="Name of the company")
    conference_call_date: str = Field(..., description="Date of the conference call")
    management_participants: List[Participant] = Field(
        ..., description="List of management participants with their designations"
    )

# --- Prompt Template ---

# Defines the prompt template for the LLM. It instructs the model to act
# as an information extractor and provides a clear example of the desired JSON format.
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are an information extractor. From the given text, identify and extract the following details:
    
    1. Company name
    2. Conference call date
    3. Management participants (with their names and designations)

    Text: {text}
    
    Respond in JSON format following this schema:
    {{
        "company_name": "<string>",
        "conference_call_date": "<DD-MM-YYYY>",
        "management_participants": [
            {{
                "name": "<string>",
                "designation": "<string>"
            }}
        ]
    }}
    """
)

# --- Error Handling and Parsing ---

# Default dictionary to be returned if the LLM fails to produce valid JSON.
# This ensures the application does not crash and always returns a predictable structure.
default_dict = {
    "company_name": "No data",
    "conference_call_date": datetime.now().date().strftime("%d-%m-%Y"),
    "management_participants": [
        {
            "name": "No data",
            "designation": "No data"
        }
    ]
}

# A robust function to safely parse the LLM's response.
def safe_parse(response: str, model: BaseModel, default: dict) -> dict:
    """Parses a JSON string into a Pydantic model, with robust error handling.

    Args:
        response: The raw string response from the LLM.
        model: The Pydantic model to validate against.
        default: The default dictionary to return on parsing failure.

    Returns:
        A dictionary representing the parsed data or the default dictionary.
    """
    try:
        # Use Pydantic's `model_validate_json` for parsing and validation.
        return model.model_validate_json(response)
    except json.JSONDecodeError as e:
        # Catches cases where the response is not valid JSON.
        return default
    except Exception:
        # Catches other potential validation or parsing errors.
        return default

# --- LangChain Chain Definition ---

# The core LangChain pipeline is constructed using LCEL.
# It chains the prompt, the LLM, a string output parser, and the safe parsing function.
extractorchain = prompt | llm_groq | StrOutputParser() | RunnableLambda(
    lambda r: safe_parse(r, ConferenceCall, default_dict)
)

# --- Main Function ---

def meta_data(docs: list) -> dict:
    """Extracts metadata from a list of documents.

    This function concatenates the content of the first two documents (assuming
    metadata is at the beginning of the transcript), invokes the extraction
    chain, and returns the result as a standard Python dictionary.

    Args:
        docs: A list of document objects, typically representing the pages of a PDF.

    Returns:
        A dictionary containing the extracted metadata.
    """
    # Concatenate the first two document pages, which typically contain the metadata.
    combined_text = docs[0].page_content + docs[1].page_content
    
    # Invoke the pre-defined extraction chain with the combined text.
    result = extractorchain.invoke({
        "text": combined_text
    })

    # Convert the Pydantic model output to a JSON string and then to a dictionary.
    result_json = result.model_dump_json(indent=2)
    return json.loads(result_json)