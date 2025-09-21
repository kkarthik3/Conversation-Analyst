from langchain_community.vectorstores import FAISS
from api.embedding import embeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from api.llm import groq_chat,chat_oss

# Prompt for RAG
rag_qa_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the retrieved documents to answer the question.

Question:
{question}

Retrieved documents:
{documents}

Instructions:
- Provide a **structured answer** in short bullet points or paragraphs, like:
- Answer **directly and concisely**, suitable for reporting or investor notes.
- You can use aother documents answer which id=s directky related to the given context
- Do **not** start with phrases like "According to the documents" or "Based on the retrieved documents."
- Use **only information from the documents**.
- If there is absolutely no relevant information, respond with:
"I’m not aware of that."

Please answer the question.
""")

parser = StrOutputParser()

def rag_pipeline(user_input):
    # Load FAISS index
    store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    
    # Retrieve top 10 docs with scores
    docs_with_scores = store.similarity_search_with_score(user_input, k=10)
    # Sort descending by score (highest similarity first)
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Prepare docs text with scores included
    docs_text = "\n\n".join([
        f"[Score: {s:.2f}] {d.page_content} | Metadata: {d.metadata}" 
        for d, s in docs_with_scores
    ])

    # Build chain
    extractorchain = rag_qa_prompt | chat_oss | parser
    result = extractorchain.invoke({
        "question": user_input, 
        "documents": docs_text
    })

    # Return the result and the top docs with their scores
    return result, docs_with_scores
