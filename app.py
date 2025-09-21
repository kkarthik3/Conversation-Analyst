"""STREAMLIT APPLICATION FOR EARNINGS CALL ANALYSIS.

This script sets up a multi-page Streamlit application to analyze earnings
call transcripts. It provides features for loading and processing PDF
transcripts, extracting metadata and key topics, summarizing content, and
interacting with a RAG-powered AI assistant for Q&A.
"""

import streamlit as st
from utils.doc_parser import file_load, Pattern_extract, split_docs_into_sections, group_by_pattern
from utils.metadata import meta_data
import os
from streamlit_option_menu import option_menu
from utils.create_topics import extract_topics
from utils.create_summary import summarizer
from concurrent.futures import ThreadPoolExecutor
from api.embedding import create_store
from utils.rag import rag_pipeline
from typing import List, Dict, Any


# Configure the Streamlit page layout and title.
st.set_page_config(
    page_title="Earnings Call Analyzer",
    layout="wide"
)

# Initialize a thread pool executor for background tasks like creating the vector store.
executor = ThreadPoolExecutor(max_workers=2)

# Initialize session state variables to maintain data across user interactions.
# This prevents re-running expensive functions every time the page reloads.
if "data" not in st.session_state:
    st.session_state["data"] = "Q2FY24_LaurusLabs_EarningsCallTranscript.pdf"
if "docs" not in st.session_state:
    st.session_state.docs = []
if "load_flag" not in st.session_state:
    st.session_state.load_flag = False
if "remarks_topics" not in st.session_state:
    st.session_state.remarks_topics = {"topics":[]}
if "QA_topics" not in st.session_state:
    st.session_state.QA_topics= {"topics":[]}
if "remarks" not in st.session_state:
    st.session_state.remarks = []
if "QA" not in st.session_state:
    st.session_state.QA = []

# Define and create a directory for temporary file uploads.
upload_folder = "temp_uploads"
os.makedirs(upload_folder, exist_ok=True)
    

# Create a horizontal option menu for navigation.
option = option_menu(
    None, 
    ["Load Data", "Opening Remarks", 'Q&A Session', "AI Assistant"],
    icons=['upload', "journals", 'patch-question', "robot"],
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal",
)

# --- Helper Function ---

def meta(docs: List) -> Dict[str, Any]:
    """Wrapper function to extract metadata from documents.
    
    Args:
        docs: A list of document objects.

    Returns:
        A dictionary containing extracted metadata.
    """
    return meta_data(docs)

# --- UI: Page Logic based on Menu Selection ---

if option == "Load Data":
    st.title("Earnings Call Analyzer")
    st.markdown("AI-powered transcript analysis with topic extraction and summarization")

    st.header("Get Started")
    st.markdown("Choose an option to begin analyzing an earnings call transcript:")

    # Show data loading options if a file has not been processed yet.
    if not st.session_state.load_flag:
        col1, col2 = st.columns(2)

        # Demo Transcript Column.
        with col1:
            st.subheader("Demo Transcript ➡️")
            st.write("Try the application with our sample Laurus Labs earnings call.")

            if st.button("Load Demo Transcript", use_container_width=True):
                with st.spinner("Processing demo transcript... This may take a few minutes."):
                    st.session_state.docs = file_load(st.session_state["data"], True)
                    st.session_state.load_flag = True
                st.success("Demo transcript loaded!")
                st.rerun()  # Refresh the page to show the results.

        # Upload Custom File Column.
        with col2:
            st.subheader("Upload Custom File")
            st.write("Upload an earnings call transcript (PDF format):")

            uploaded_file = st.file_uploader(
                "Drag and drop file here",
                type=["pdf"],
                help="Limit 200MB per file",
                key="file_uploader"
            )

            if uploaded_file is not None:
                st.success("File uploaded successfully!")
                
                # Save the uploaded file to a temporary location.
                tmp_path = os.path.join(upload_folder, uploaded_file.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Processing uploaded file... This may take a few minutes."):
                    st.session_state.docs = file_load(tmp_path, False)
                    st.session_state.load_flag = True
                
                # Clean up the temporary file after processing.
                os.remove(tmp_path)
                st.rerun()

    # Display metadata and an overview once the file is processed.
    else:
        st.success("Transcript processed successfully!")

        # Extract and display key metadata in a three-column layout.
        col1, col2, col3 = st.columns(3)
        meta_info = meta_data(st.session_state.docs)

        # Column 1: Company
        with col1:
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #2e3b4d; text-align: center; margin-right: 10px;">
                    <h3 style="color: #ffffff;">Company</h3>
                    <p style="color: #cccccc; font-size: 24px;">{meta_info["company_name"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Column 2: Call Date
        with col2:
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #2e3b4d; text-align: center; margin-right: 10px;">
                    <h3 style="color: #ffffff;">Call Date</h3>
                    <p style="color: #cccccc; font-size: 24px;">{meta_info["conference_call_date"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Column 3: Pages Processed
        with col3:
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #2e3b4d; text-align: center;">
                    <h3 style="color: #ffffff;">Pages Processed</h3>
                    <p style="color: #cccccc; font-size: 24px;">{len(st.session_state.docs)} pages</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Content overview")
        
        # Process the documents to extract dialogues and segment them.
        particpants = [participant['name'] for participant in meta_info['management_participants']]
        st.session_state.pattern = Pattern_extract(docs=st.session_state.docs, management=particpants)
        st.session_state.remarks, st.session_state.QA = split_docs_into_sections(st.session_state.pattern)

        # Asynchronously create the FAISS vector store for the RAG pipeline.
        executor.submit(create_store, st.session_state.remarks + st.session_state.QA)

        # Display counts of remarks and Q&A chunks.
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #2e3b4d; text-align: center;">
                    <h3 style="color: #ffffff;">Opening Remarks</h3>
                    <p style="color: #cccccc; font-size: 24px;">{len(st.session_state.remarks)} Chunks</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #2e3b4d; text-align: center;">
                    <h3 style="color: #ffffff;">Q&A Session</h3>
                    <p style="color: #cccccc; font-size: 24px;">{len(st.session_state.QA)} Chunks</p>
                </div>
                """,
                unsafe_allow_html=True
            )

elif option == "Opening Remarks":
    st.header("Opening Remarks")
    st.divider()
    tab1, tab2, tab3 = st.tabs(["View Chunks", "Generate Topics", "Create Summary"])

    with tab1:
        st.subheader("Raw Text - Who Said What")
        for index, remarks in enumerate(st.session_state.remarks):
            with st.expander(f"C{index+1}  {remarks.metadata['speaker']}"):
                st.markdown(remarks.page_content)

    with tab2:
        if st.button("Click here to Generate Topics") or st.session_state.remarks_topics["topics"]:
            with st.spinner("Please wait until Topics are loaded..."):
                st.session_state.remarks_topics = extract_topics(st.session_state.remarks)
            for index, topic in enumerate(st.session_state.remarks_topics["topics"]):
                st.success(f"Topic {index+1}  -->  {topic}")

    with tab3:
        if st.session_state.remarks_topics["topics"]: 
            st.subheader("Summaries for Generated Topics")

            @st.fragment
            def topic_summary():
                """Allows users to select topics and generate a summary."""
                with st.container(border=True):
                    selected_options = st.multiselect(
                        "Select topics to summarize:",
                        options=st.session_state.remarks_topics["topics"],
                        default=st.session_state.remarks_topics["topics"]
                    )
                submit_remark = st.button("Click to Proceed to generate summary")
                
                if submit_remark:
                    with st.spinner("Generating summary..."):
                        summary_text = summarizer(
                            input="\n".join([doc.page_content for doc in st.session_state.remarks]),
                            topics=", ".join(selected_options)
                        )
                    st.success(summary_text)

            topic_summary()
        else:
            st.warning("Please Generate topics first.")
            
elif option == "Q&A Session":
    st.header("Q&A Session")
    st.divider()
    tab1, tab2, tab3 = st.tabs(["View Chunks", "Generate Topics", "Create Summary"])
    qa_batches = group_by_pattern(st.session_state.QA)

    with tab1:
        st.subheader("Batched Q&A")
        for idx, (questions, answers) in enumerate(qa_batches, start=1):
            with st.container():
                with st.expander(f"## Question {idx}"):
                    if questions:
                        st.markdown("**Questions:**")
                        for q in questions:
                            st.markdown(f"- *{q.metadata['speaker']}*: {q.page_content}")
                    if answers:
                        st.markdown("**Answers:**")
                        for a in answers:
                            st.markdown(f"- *{a.metadata['speaker']}*: {a.page_content}")

    with tab2:
        if st.button("Click here to Generate Topics") or st.session_state.QA_topics["topics"]:
            with st.spinner("Please wait until Topics are loaded..."):
                st.session_state.QA_topics = extract_topics(st.session_state.QA)
            for index, topic in enumerate(st.session_state.QA_topics["topics"]):
                st.success(f"Topic {index+1}  -->  {topic}")

    with tab3:
        if st.session_state.QA_topics["topics"]: 
            st.subheader("Summaries for Generated Topics")

            @st.fragment
            def topic_summary():
                """Allows users to select topics and generate a summary for the Q&A section."""
                with st.container(border=True):
                    selected_options = st.multiselect(
                        "Select topics to summarize:",
                        options=st.session_state.QA_topics["topics"],
                        default=st.session_state.QA_topics["topics"]
                    )
                submit_remark = st.button("Click to Proceed to generate summary")

                if submit_remark:
                    with st.spinner("Generating summary..."):
                        summary_text = summarizer(
                            input="\n".join([doc.page_content for doc in st.session_state.QA]),
                            topics=", ".join(selected_options)
                        )
                    st.success(summary_text)
            
            topic_summary()
        else:
            st.warning("Please Generate topics first.")

elif option == "AI Assistant":
    if st.session_state.docs:
        # Chat input for the user's question.
        user_input = st.chat_input("Ask me a question")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result, details = rag_pipeline(user_input)
                    st.markdown(result)

                with st.expander("Metadata"):
                    for doc, score in details:
                        rag_meta = doc.metadata
                        line = (
                            f"Speaker: {rag_meta.get('speaker', 'N/A')} | "
                            f"Type: {rag_meta.get('type', 'N/A')} | "
                            f"Role: {rag_meta.get('role', 'N/A')} | "
                            f"Similarity: {score:.4f} |\n"
                            f"Content: {doc.page_content[:100]}.......\n"
                        )
                        st.markdown(line)
    else:
        st.warning("Please load your documents first in the 'Load Data' tab to use the AI Assistant.")