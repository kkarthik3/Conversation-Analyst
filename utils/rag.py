"""RETRIEVAL-AUGMENTED GENERATION (RAG) PIPELINE.

This module implements a RAG pipeline for question-answering based on
a pre-built FAISS vector store. It retrieves relevant documents,
formats them, and then uses a large language model to generate a
concise, structured answer based on the retrieved information.
"""


from langchain_community.vectorstores import FAISS
from api.embedding import embeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from api.llm import groq_chat, chat_oss
from typing import List, Tuple
from langchain.schema import Document


# PromptTemplate for the RAG task.
rag_qa_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the retrieved documents to answer the question.

Question:
{question}

Retrieved documents:
{documents}

Instructions:
- Provide a **structured answer** in short bullet points or paragraphs, like:
- Answer **directly and concisely**, suitable for reporting or investor notes.
- You can use another document's answer which is directly related to the given context.
- Do **not** start with phrases like "According to the documents" or "Based on the retrieved documents."
- Use **only information from the documents**.
- If there is absolutely no relevant information, respond with:
"I’m not aware of that."

Please answer the question.
""")

# Define an output parser that converts the LLM's response into a simple string.
parser = StrOutputParser()


def rag_pipeline(user_input: str) -> Tuple[str, List[Tuple[Document, float]]]:
    """Executes the RAG pipeline to answer a user's question.

    This function performs the following steps:
    1. Loads a local FAISS vector store.
    2. Searches for documents most similar to the user's question.
    3. Sorts the retrieved documents by their similarity score.
    4. Formats the documents with their scores for the LLM prompt.
    5. Invokes a LangChain pipeline to generate an answer.
    6. Returns the generated answer and the list of retrieved documents with scores.

    Args:
        user_input: The question string from the user.

    Returns:
        A tuple containing:
        - The generated answer as a string.
        - A list of tuples, where each tuple contains a LangChain Document
          and its similarity score, sorted in descending order.
    """
    #  Load the FAISS vector store from the local directory.
    # The `allow_dangerous_deserialization=True` flag is required because
    # the index may contain custom data types. Use with caution.
    store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    
    #  Retrieve the top 10 documents with their similarity scores.
    docs_with_scores = store.similarity_search_with_score(user_input, k=10)
    
    #  Sort the documents by score in descending order (highest similarity first).
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    #  Prepare a formatted string of the retrieved documents.
    # This string is what gets passed to the LLM in the prompt. It includes
    # the score and metadata for each document for full context.
    docs_text = "\n\n".join([
        f"[Score: {s:.2f}] {d.page_content} | Metadata: {d.metadata}" 
        for d, s in docs_with_scores
    ])

    #  Build and invoke the LangChain pipeline.
    # The chain first formats the prompt with the question and documents,
    # then passes it to the `chat_oss` LLM, and finally parses the output.
    extractorchain = rag_qa_prompt | chat_oss | parser
    result = extractorchain.invoke({
        "question": user_input, 
        "documents": docs_text
    })

    #  Return the generated answer and the top documents with scores.
    return result, docs_with_scores