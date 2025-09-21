import streamlit as st
from utils.doc_parser import file_load, Pattern_extract,split_docs_into_sections, group_by_pattern
from utils.metadata import meta_data
import os
from streamlit_option_menu import option_menu
from utils.create_topics import extract_topics
from utils.create_summary import summarizer
from concurrent.futures import ThreadPoolExecutor
from api.embedding import create_store
from utils.rag import rag_pipeline



# --- Page Configuration ---
st.set_page_config(
    page_title="Earnings Call Analyzer",
    layout="wide"
)

executor = ThreadPoolExecutor(max_workers=2)

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
    st.session_state.remarks =[]


upload_folder = "temp_uploads"
os.makedirs(upload_folder, exist_ok=True)
    


option = option_menu(None, ["Load Data", "Opening Remarks", 'Q&A Session',"AI Assistant"],
                     icons=['upload', "journals", 'patch-question',"robot"],
                     menu_icon="cast", default_index=0, orientation="horizontal",)

def meta(docs):
    return meta_data(docs)

if option == "Load Data":
    st.title("Earnings Call Analyzer")
    st.markdown("AI-powered transcript analysis with topic extraction and summarization")

    st.header("Get Started")
    st.markdown("Choose an option to begin analyzing an earnings call transcript:")

    # Show columns only if data hasn't been loaded yet
    if st.session_state.load_flag == False:
        col1, col2 = st.columns(2)

        # --- Demo Transcript Column ---
        with col1:
            st.subheader("Demo Transcript ➡️")
            st.write("Try the application with our sample Laurus Labs earnings call.")

            if st.button("Load Demo Transcript", use_container_width=True):
                with st.spinner("Processing demo transcript... This may take a few minutes."):
                    st.session_state.docs = file_load(st.session_state["data"], True)
                    st.session_state.load_flag = True
                    st.success("Demo transcript loaded!")

                    st.rerun()  # Refresh the page to hide columns

        # --- Upload Custom File Column ---
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

                tmp_path = os.path.join(upload_folder, uploaded_file.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Processing uploaded file... This may take a few minutes."):
                    st.session_state.docs = file_load(tmp_path, False)
                    st.session_state.load_flag = True
                os.remove(tmp_path)

                st.rerun()  # Refresh the page to hide columns

    else:
        st.success("Transcript processed successfully!")

        # Use st.columns to create a 3-column layout as shown in the image
        col1, col2, col3 = st.columns(3)

        # Assuming meta_data function extracts the relevant info from the processed docs
        # Note: You'll need to implement the meta_data function to extract this data
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
        particpants = [participant['name'] for participant in meta_info['management_participants']]
        st.session_state.pattern = Pattern_extract(docs = st.session_state.docs, management=particpants)
        st.session_state.remarks, st.session_state.QA = split_docs_into_sections(st.session_state.pattern)

        executor.submit(create_store(st.session_state.remarks + st.session_state.QA)) ##Pdf Loading for RAG

        col1,col2 = st.columns(2)
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


if option == "Opening Remarks":
    st.header("Opening Remarks")
    st.divider()
    tab1, tab2, tab3 = st.tabs(["View Chunks", "Generate Topics", "Create Summary"])

    with tab1:
        st.subheader("Raw Text who said What")

        for index, remarks in enumerate(st.session_state.remarks):
            with st.expander(f"C{index+1}  {remarks.metadata["speaker"]}"):
                st.markdown(remarks.page_content)

    with tab2:

        if st.button("Click here to Genetate Topics") or st.session_state.remarks_topics["topics"]:
            with st.spinner("Please wait Untill Topic is loaded"):
                st.session_state.remarks_topics = extract_topics(st.session_state.remarks)
            for index,topic in enumerate(st.session_state.remarks_topics["topics"]):
                st.success(f"Topic {index+1}  -->  {topic}")

    with tab3:
        if st.session_state.remarks_topics["topics"] : 
            st.subheader("Summaries for Genrated topics")

            if st.session_state.remarks_topics["topics"]:
                @st.fragment
                def topic_summary():
                    with st.container(border=True):
                        selected_options = []
                        for option in st.session_state.remarks_topics["topics"]:
                            if st.checkbox(option):
                                selected_options.append(option)
                        submit_remark = st.button("Click Proceed To generate summary")

                    if submit_remark:
                        st.success(summarizer(input= st.session_state.remarks,
                                              topics=selected_options))
                
                topic_summary()

        else:
            st.warning("Please Genarate topics first")

            
if option == "Q&A Session":
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

        if st.button("Click here to Genetate Topics") or st.session_state.QA_topics["topics"]:
            with st.spinner("Please wait Untill Topic is loaded"):
                st.session_state.QA_topics = extract_topics(st.session_state.QA)
            for index,topic in enumerate(st.session_state.QA_topics["topics"]):
                st.success(f"Topic {index+1}  -->  {topic}")

    with tab3:
        if st.session_state.QA_topics["topics"] : 
            st.subheader("Summaries for Genrated topics")

            if st.session_state.QA_topics["topics"]:
                @st.fragment
                def topic_summary():
                    with st.container(border=True):
                        selected_options = []
                        for option in st.session_state.QA_topics["topics"]:
                            if st.checkbox(option):
                                selected_options.append(option)
                        submit_remark = st.button("Click Proceed To generate summary")

                    if submit_remark:
                        st.success(summarizer(input= st.session_state.remarks,
                                              topics=selected_options))
                
                topic_summary()

        else:
            st.warning("Please Genarate topics first")

if option=="AI Assistant":
    if st.session_state.docs:
        if input:= st.chat_input("Ask me a question"):
            with st.chat_message("user"):
                st.markdown(input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result, details = rag_pipeline(input)
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
        st.warning("Load your Docs first")