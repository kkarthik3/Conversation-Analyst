from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
import fitz
import os
import tempfile
import shutil
import os
import streamlit as st

# Get the current working directory
current_dir = os.getcwd()

# Build relative path (assuming the PDF is in the same folder as the script)
file_path = os.path.join(current_dir, "Q2FY24_LaurusLabs_EarningsCallTranscript.pdf")
text_splitter = RecursiveCharacterTextSplitter()

@st.cache_data()
def file_load(path: str = file_path, default: bool = True):
    doc = fitz.open(path)

    for page in doc:
        # Define header and footer areas
        header_rect = fitz.Rect(0, 0, page.rect.width, 50)
        footer_rect = fitz.Rect(0, page.rect.height - 50, page.rect.width, page.rect.height)

        # Add redaction annotations
        page.add_redact_annot(header_rect, fill=(1, 1, 1))
        page.add_redact_annot(footer_rect, fill=(1, 1, 1))

        # Apply redaction immediately
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)  # remove text/images in the redaction areas

    # Save using a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "temp.pdf")
        doc.save(tmp_path, garbage=4, deflate=True, clean=True)
        doc.close()
        shutil.copy(tmp_path, path)

    # Load PDF into LangChain
    loader = PyMuPDFLoader(file_path if default else path)
    data = loader.load()
    return data

def Pattern_extract(docs:list, management: list):
    text = "\n".join(i.page_content.strip() for i in docs)
    
    start_phrases = [
    r"Ladies and gentlemen, good day",
    r"Good morning, everyone",
    r"Welcome to the .* conference call",
    ]

    # Combine into a single regex
    start_pattern = re.compile("|".join(start_phrases), re.I)

    match = start_pattern.search(text)
    if match:
        conversation_text = text[match.start():]  # everything from the first greeting onward
    else:
        conversation_text = text  # fallback: use entire text

    # Step 2: Regex for dialogues → assumes 'Speaker Name:' format
    dialogue_pattern = re.compile(
            r"([A-Z][A-Za-z .'-]+):\s+(.*?)(?=(?:[A-Z][A-Za-z .'-]+:)|\Z)", 
            re.S
        )

    matches = dialogue_pattern.findall(conversation_text)

    # Step 3: Build LangChain Documents
    docs = []
    for i, (speaker, speech) in enumerate(matches, start=1):
        doc = Document(
            page_content=speech.strip(),
            metadata={
                "speaker": speaker.strip(),
                "order": i,
                "type":"Answerer" if speaker.strip() in management else "Questioner",
                "role" : "Management" if speaker.strip() in management else "Investor"
            }
        )
        docs.append(doc)

    return docs

def split_docs_into_sections(extract_docs):
    remarks = []
    action_qa = []
    final = []


    qa_start_pattern = re.compile(r"The first question", re.I)

    section = "Remarks"

    for doc in extract_docs:
        speaker = doc.metadata.get("speaker", "").strip()
        content = doc.page_content.strip()


        # Check if Q&A starts
        if section == "Remarks" and qa_start_pattern.search(content):
            section = "Action Q&A"

        # Append to the right section
        if section == "Remarks":
            if speaker.lower() != "moderator":
                doc.metadata["Section"] = "Remark"
                remarks.append(doc)
        elif section == "Action Q&A":
            if speaker.lower() != "moderator":
                doc.metadata["Section"] = "Q&A"
                action_qa.append(doc)

    return remarks, action_qa

def group_by_pattern(qa_docs):
    """
    Groups into batches where each batch = consecutive Question(s) followed by consecutive Answer(s).
    Example sequence: Q Q A A Q A A → [QQAA], [QAA]
    """
    batches = []
    current_qs = []
    current_as = []

    for doc in qa_docs:
        t = "Q" if doc.metadata["type"] == "Questioner" else "A"

        if t == "Q":
            # If we already have answers, close the batch
            if current_as:
                batches.append((current_qs, current_as))
                current_qs, current_as = [], []
            current_qs.append(doc)

        elif t == "A":
            current_as.append(doc)

    # Push last batch
    if current_qs or current_as:
        batches.append((current_qs, current_as))

    return batches
