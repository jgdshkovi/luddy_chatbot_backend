import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import time

# === PATH TO FOLDER ===
folder_path = "Final_scraped_txts"  

# === LOAD TEXT FILES ===
def load_documents_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
    return documents

# === SPLIT INTO CHUNKS ===
def split_documents(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# === EMBEDDINGS SETUP ===
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === SAVE TO FAISS ===
def save_to_faiss(split_docs, embeddings, index_path="faiss_index_luddy"):
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(index_path)
    print(f"✅ FAISS index saved to '{index_path}'")

start_time = time.time()

docs = load_documents_from_folder(folder_path)
print(f"📄 Loaded {len(docs)} documents.")

for doc in docs:
    web_url = ""
    web_url = doc.page_content[:400].split('\n')[0][14:]
    doc.metadata['webURL'] = web_url

split_docs = split_documents(docs)
print(f"Split into {len(split_docs)} chunks.")

embeddings = get_embedding_model()
save_to_faiss(split_docs, embeddings)

elapsed_time = start_time- time.time()
print(f"Elapsed Time: {elapsed_time:.2f} seconds")