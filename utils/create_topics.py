from typing import List
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import json

from api.llm import llm_oss , llm_groq # your initialized LLM instance

# Step 1: Define Pydantic model for Topics
class TopicsOutput(BaseModel):
    topics: List[str] = Field(..., description="List of relevant topics extracted from context")

# Step 2: Prompt template with sample output
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are an assistant that extracts Top 5 reelevant key topics from the given context which is relevant.
    
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

# Step 3: Default dictionary in case parsing fails
default_dict = {
    "topics": []
}

# Step 4: Safe parse function
def safe_parse(response: str, model, default: dict):
    try:
        return model.model_validate_json(response)
    except json.JSONDecodeError:
        return default
    except Exception:
        return default

# Step 5: Build extraction chain
extractorchain = prompt | llm_oss | StrOutputParser() | RunnableLambda(
    lambda r: safe_parse(r, TopicsOutput, default_dict)
)

# Step 6: Function to get topics from documents
def extract_topics(docs):
    """
    docs: list of document objects with .page_content attribute
    """
    combined_text = " ".join([doc.page_content for doc in docs])
    
    result = extractorchain.invoke({
        "text": combined_text
    })

    # Convert to JSON string and then back to dict
    result_json = result.model_dump_json(indent=2)
    return json.loads(result_json)