from typing import List
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from pydantic import BaseModel, Field
from datetime import date,datetime
import json

from api.llm import llm_groq

class Participant(BaseModel):
    name: str = Field(..., description="Full name of the management participant")
    designation: str = Field(..., description="Designation of the participant")

class ConferenceCall(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    conference_call_date: str = Field(..., description="Date of the conference call")
    management_participants: List[Participant] = Field(
        ..., description="List of management participants with designations"
    )


from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are an information extractor. From the given text, identify and extract the following details:
    
    1. Company name
    2. Conference call date
    3. Management participants (with their names and designations) Include all even Moderator

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

def safe_parse(response: str, model, default: dict):
    try:
        return model.model_validate_json(response)
    except json.JSONDecodeError as e:
        return default

extractorchain = prompt | llm_groq | StrOutputParser() | RunnableLambda(
    lambda r: safe_parse(r, ConferenceCall, default_dict)
)


def meta_data(docs) :


    result = extractorchain.invoke({
        "text": docs[0].page_content + docs[1].page_content
    })

    # result = result.model_dump_json(indent=2)

    return result