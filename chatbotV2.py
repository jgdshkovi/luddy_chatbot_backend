import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from groq import Groq


# Serve Model
from fastapi import FastAPI, Request
from pydantic import BaseModel

GROQ_API_KEY = "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"

# === Initializing re-ranker, embedding models and FAISS ===
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index_luddy", embedding_model, allow_dangerous_deserialization = 'True')

# === Function to perform hybrid search ===
def hybrid_search(query, top_k=20):
    semantic_results = db.similarity_search_with_score(query, k=top_k * 2)
    keyword_filtered = [(doc, score) for doc, score in semantic_results if query.lower() in doc.page_content.lower()]
    if len(keyword_filtered) >= top_k:
        return keyword_filtered[:top_k]
    else:
        extra_needed = top_k - len(keyword_filtered)
        additional = [item for item in semantic_results if item not in keyword_filtered]
        return keyword_filtered + additional[:extra_needed]

def format_chat_history(chat_history, max_turns=2):
    messages = chat_history.messages
    if len(messages) > max_turns * 2:
        summary_input = "\n".join([f"{'User' if m.type == 'human' else 'LuddyBot'}: {m.content}" for m in messages[:-max_turns*2]])
        summary_prompt = f"Summarize the following conversation between a student and LuddyBot:\n\n{summary_input}"
        
        client = Groq(api_key=GROQ_API_KEY)
        summary = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": summary_prompt}]
        ).choices[0].message.content.strip()
        
        recent_turns = messages[-max_turns*2:]
        recent_text = "\n".join([f"{'User' if m.type == 'human' else 'LuddyBot'}: {m.content}" for m in recent_turns])
        
        return f"Summary of earlier conversation:\n{summary}\n\nRecent Chat:\n{recent_text}"
    else:
        return "\n".join([f"{'User' if m.type == 'human' else 'LuddyBot'}: {m.content}" for m in messages])


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

# === Final answer generation function ===
def generate_llama_answer(query, session_id, top_k=5):
    # === Retrieve Chat History ===
    chat_history = get_session_history(session_id)
    
    # === Hybrid + Reranking ===
    hybrid_results = hybrid_search(query, top_k=20)
    pairs = [(query, doc.page_content) for doc, _ in hybrid_results]
    scores = reranker_model.predict(pairs)
    reranked = sorted(zip([doc for doc, _ in hybrid_results], scores), key=lambda x: x[1], reverse=True)
    top_docs = reranked[:top_k]
    context = "\n\n".join([f"Source: {doc.metadata.get('webURL', 'N/A')} \n{doc.page_content}"   for doc, _ in top_docs])

    # === Chat History Formatting ===
    chat_context = format_chat_history(chat_history, 2)

    print("Context******************\n",chat_context)

    # === Prompt Assembly ===
    prompt = f""" You are LuddyBot, a helpful AI assistant at IU Luddy School.
                Answer the following question in a clear and structured format based on the provided context and chat_history. Please also provide with the necessary website links. Don't mention "context" or "based on context" in the response.
                Previous Chat: {chat_context}
                Context: {context}
                Student's Question: {query}
                LuddyBot's Structured Answer:
                """

    # === LLaMA Inference ===
    client = Groq(api_key=GROQ_API_KEY)
    chat_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = chat_response.choices[0].message.content.strip()

    # === Save Messages ===
    chat_history.add_user_message(query)
    chat_history.add_ai_message(answer)

    return answer





app = FastAPI()

# Define input format
class Query(BaseModel):
    prompt: str
    session_id: str


@app.post("/ask")
async def ask_llm(query: Query):
    response = generate_llama_answer(
        query=query.prompt,
        session_id=query.session_id
    )
    return {"response": response}

