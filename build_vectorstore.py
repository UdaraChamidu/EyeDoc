import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vectorstore(pdf_path: str, save_path: str = "vectorstore/eye_faiss"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

    print("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_documents(documents)

    print("🔍 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

    print("📦 Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    print(f"💾 Saving to {save_path} ...")
    vectorstore.save_local(save_path)

    print("✅ Vectorstore built and saved successfully!")
    return vectorstore
