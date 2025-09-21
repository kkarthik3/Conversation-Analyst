"""PDF DOCUMENT PROCESSING FOR EARNINGS CALL TRANSCRIPTS.

This script provides a comprehensive pipeline for processing earnings call
PDFs. It includes functions for loading and preprocessing PDF files,
extracting and structuring conversation text, and segmenting the content
into logical sections like "Remarks" and "Q&A" for further analysis.
"""

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
from typing import List, Tuple


# Get the current working directory.
current_dir = os.getcwd()

# Build relative path to the PDF file.
# Assumes the PDF is located in the same directory as the script.
file_path = os.path.join(current_dir, "Q2FY24_LaurusLabs_EarningsCallTranscript.pdf")
text_splitter = RecursiveCharacterTextSplitter()

@st.cache_data()
def file_load(path: str = file_path, default: bool = True) -> List[Document]:
    """Loads a PDF file, redacts headers/footers, and returns a list of LangChain Documents.

    This function uses PyMuPDF (fitz) to open the PDF, apply redaction to
    the header and footer areas to remove boilerplate text, and then saves
    the modified file to a temporary location. Finally, it uses
    PyMuPDFLoader to load the content into LangChain's Document format.

    Args:
        path: The file path to the PDF to be loaded.
        default: A boolean to indicate whether to use the default file path
                 or the path provided in the argument.

    Returns:
        A list of LangChain Document objects, where each object represents a
        page from the processed PDF.
    """
    doc = fitz.open(path)

    for page in doc:
        # Define header and footer areas for redaction.
        header_rect = fitz.Rect(0, 0, page.rect.width, 50)
        footer_rect = fitz.Rect(0, page.rect.height - 50, page.rect.width, page.rect.height)

        # Add redaction annotations to the defined areas.
        page.add_redact_annot(header_rect, fill=(1, 1, 1))
        page.add_redact_annot(footer_rect, fill=(1, 1, 1))

        # Apply the redaction immediately, removing text and images.
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    # Save the processed PDF to a temporary directory to avoid overwriting
    # the original file during processing.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "temp.pdf")
        doc.save(tmp_path, garbage=4, deflate=True, clean=True)
        doc.close()
        
        # Copy the temporary, processed file back to the original path.
        shutil.copy(tmp_path, path)

    # Load the processed PDF content into LangChain Documents.
    loader = PyMuPDFLoader(file_path if default else path)
    data = loader.load()
    return data

def Pattern_extract(docs: List[Document], management: List[str]) -> List[Document]:
    """Extracts and structures dialogues from a list of documents.

    This function combines the text from all documents, identifies the start
    of the main conversation, and then uses a regex pattern to parse the text
    into individual speaker-dialogue pairs. It then creates new LangChain
    Document objects for each dialogue, adding relevant metadata.

    Args:
        docs: A list of LangChain Document objects from the raw PDF load.
        management: A list of speaker names (strings) that belong to
                    the management team.

    Returns:
        A new list of LangChain Document objects, each representing a single
        dialogue with speaker, order, and role metadata.
    """
    # Join all document page content into a single string.
    text = "\n".join(i.page_content.strip() for i in docs)

    # Step 1: Find the start of the conversation using common greeting phrases.
    start_phrases = [
        r"Ladies and gentlemen, good day",
        r"Good morning, everyone",
        r"Welcome to the .* conference call",
    ]

    # Combine into a single regex pattern for efficient searching.
    start_pattern = re.compile("|".join(start_phrases), re.I)

    match = start_pattern.search(text)
    if match:
        conversation_text = text[match.start():]  # Everything from the first greeting onward.
    else:
        conversation_text = text  # Fallback: use the entire text if no match is found.

    # Step 2: Regex for dialogues. It assumes a "Speaker Name: dialogue" format.
    # It captures the speaker's name and the subsequent speech until the next
    # speaker name or the end of the text.
    dialogue_pattern = re.compile(
        r"([A-Z][A-Za-z .'-]+):\s+(.*?)(?=(?:[A-Z][A-Za-z .'-]+:)|\Z)",
        re.S
    )

    matches = dialogue_pattern.findall(conversation_text)

    # Step 3: Build new LangChain Documents from the parsed dialogues.
    docs = []
    for i, (speaker, speech) in enumerate(matches, start=1):
        # Create a new Document for each dialogue.
        doc = Document(
            page_content=speech.strip(),
            metadata={
                "speaker": speaker.strip(),
                "order": i,
                "type": "Answerer" if speaker.strip() in management else "Questioner",
                "role": "Management" if speaker.strip() in management else "Investor"
            }
        )
        docs.append(doc)

    return docs

def split_docs_into_sections(extract_docs: List[Document]) -> Tuple[List[Document], List[Document]]:
    """Splits a list of dialogue documents into two sections: "Remarks" and "Q&A".

    The function iterates through the documents and identifies the transition
    from the initial management remarks to the Q&A session based on a
    specific phrase. It then categorizes each document into the appropriate
    section and adds a "Section" key to its metadata.

    Args:
        extract_docs: A list of LangChain Document objects, each representing
                      a dialogue.

    Returns:
        A tuple containing two lists of Document objects: (remarks, action_qa).
    """
    remarks = []
    action_qa = []

    # Regex to find the start of the Q&A section.
    qa_start_pattern = re.compile(r"The first question", re.I)

    section = "Remarks"

    for doc in extract_docs:
        speaker = doc.metadata.get("speaker", "").strip()
        content = doc.page_content.strip()

        # Check if the Q&A session has started.
        if section == "Remarks" and qa_start_pattern.search(content):
            section = "Action Q&A"

        # Append to the correct section, excluding the moderator.
        if speaker.lower() != "moderator":
            if section == "Remarks":
                doc.metadata["Section"] = "Remark"
                remarks.append(doc)
            elif section == "Action Q&A":
                doc.metadata["Section"] = "Q&A"
                action_qa.append(doc)

    return remarks, action_qa

def group_by_pattern(qa_docs: List[Document]) -> List[Tuple[List[Document], List[Document]]]:
    """Groups consecutive Q&A dialogues into logical batches.

    This function iterates through the Q&A documents and groups them into
    batches where each batch consists of one or more consecutive questions
    followed by one or more consecutive answers. This is useful for
    processing and summarizing Q&A sessions.

    Args:
        qa_docs: A list of LangChain Document objects from the Q&A section.

    Returns:
        A list of tuples. Each tuple contains two lists: the first for
        questions and the second for answers in that specific Q&A batch.
    """
    batches = []
    current_qs = []
    current_as = []

    for doc in qa_docs:
        # Determine if the dialogue is a Questioner (Q) or Answerer (A).
        doc_type = "Q" if doc.metadata["type"] == "Questioner" else "A"

        if doc_type == "Q":
            # If we have answers from the previous batch, close it and start a new one.
            if current_as:
                batches.append((current_qs, current_as))
                current_qs, current_as = [], []
            current_qs.append(doc)

        elif doc_type == "A":
            current_as.append(doc)

    # Append the last incomplete batch after the loop.
    if current_qs or current_as:
        batches.append((current_qs, current_as))

    return batches