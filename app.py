import streamlit as st
from utils.doc_parser import file_load
from utils.metadata import meta_data
import os
from streamlit_option_menu import option_menu

# --- Page Configuration ---
st.set_page_config(
    page_title="Earnings Call Analyzer",
    layout="wide"
)


if "data" not in st.session_state:
    st.session_state["data"] = "Q2FY24_LaurusLabs_EarningsCallTranscript.pdf"

if "docs" not in st.session_state:
    st.session_state.docs = []

if "load_flag" not in st.session_state:
    st.session_state.load_flag = False


upload_folder = "temp_uploads"
os.makedirs(upload_folder, exist_ok=True)
    
st.title("Earnings Call Analyzer")
st.markdown("AI-powered transcript analysis with topic extraction and summarization")

st.header("Get Started")
st.markdown("Choose an option to begin analyzing an earnings call transcript:")


option = option_menu(None, ["Load Data", "Visual Analysis", 'Words Distribution',"Reviews Authenticity","Topic Modelling"],
                     icons=['upload', "images", 'alphabet',"fingerprint","diagram-3"],
                     menu_icon="cast", default_index=0, orientation="horizontal",)

def meta(docs):
    return meta_data(docs)

if option == "Load Data":
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
                <div style="padding: 20px; border-radius: 10px; background-color: #2e3b4d; text-align: center;">
                    <h3 style="color: #ffffff;">Company</h3>
                    <p style="color: #cccccc; font-size: 24px;">{meta_info.company_name}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Column 2: Call Date
        with col2:
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #2e3b4d; text-align: center;">
                    <h3 style="color: #ffffff;">Call Date</h3>
                    <p style="color: #cccccc; font-size: 24px;">{meta_info.conference_call_date}</p>
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
