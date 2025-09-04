import os
import warnings
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# -----------------------------
# Vectorstore / Memory
# -----------------------------
class RAGAssistant:
    def __init__(self):
        self.vectorstore = None
        self.memory = None
        self.prompt_template = PromptTemplate.from_template("""
        You are an AI assistant specialized in Eye Diseases.
        Use the following context to answer.
        If question is unclear, ask for more info.
        Keep answers friendly, simple, and accurate.

        Context:
        {context}

        Question: {question}
        Answer:
        """)
        self.load_vectorstore()
        self.load_memory()

    def load_vectorstore(self):
        if self.vectorstore is None:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
            self.vectorstore = FAISS.load_local(
                "vectorstore/eye_faiss",
                embeddings,
                allow_dangerous_deserialization=True
            )

    def load_memory(self):
        if self.memory is None and self.vectorstore:
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_chain(self):
        llm = ChatGoogleGenerativeAI(
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            model="gemini-2.5-flash",
            temperature=0
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k":3}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt_template},
            return_source_documents=False,
            output_key="answer"
        )

rag_assistant = RAGAssistant()
