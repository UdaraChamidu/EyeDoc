import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Path to your medical book
pdf_path = "Kanski’s clinical ophthalmology _ a systematic approach.pdf"

# Load and split PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(documents)

# Create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# Save to disk
vectorstore.save_local("vectorstore/eye_faiss")
print("✅ Vectorstore saved to 'vectorstore/eye_faiss'")
