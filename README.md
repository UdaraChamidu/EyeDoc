# 🧿 Eye Disease Chatbot using LangChain + OpenAI (RAG-based)

This project implements a **Conversational Retrieval-Augmented Generation (RAG) chatbot** to provide answers to ophthalmology-related queries based on _Kanski’s Clinical Ophthalmology: A Systematic Approach_. It uses **LangChain**, **OpenAI's LLM**, and **ChromaDB** for embedding-based retrieval.

![Conversational RAG Chain Architecture](image.png)

---

## 🚀 Features

- Conversational chatbot for eye disease information
- Retrieval-Augmented Generation (RAG) with history-aware memory
- Uses OpenAI's GPT (via `langchain-openai`)
- Based on trusted medical content (_Kanski’s Clinical Ophthalmology_)
- Chat history awareness for improved interaction

---

## 📚 Dataset

This chatbot was built using the textbook:

> **Kanski’s Clinical Ophthalmology: A Systematic Approach**  
> Loaded as a PDF and chunked for vector search.

---

## 🛠️ Tech Stack

- **LangChain**: LLM orchestration and RAG pipeline
- **OpenAI GPT (gpt-3.5-turbo)**: Language model
- **ChromaDB**: Vector database for storing document embeddings
- **Google Colab / Jupyter Notebook**: Development environment

---

## 🧩 Architecture

The chatbot implements a **Conversational RAG Chain**:

1. **History Management**: Stores the user’s previous messages.
2. **History-Aware Retriever**: Improves retrieval using conversational history.
3. **Prompt Construction**: Combines instructions, context, history, and user query.
4. **LLM QA Chain**: OpenAI GPT model generates answers.

---

## 🧪 How It Works

### 🔹 Phase 1: Build the Basic RAG Chain

1. **Install Dependencies**:
    ```python
    !pip install langchain langchain-openai langchain-chroma langchain_community
    ```

2. **Initialize LLM**:
    ```python
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    ```

3. **Initialize Embeddings**:
    ```python
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    ```

4. **Load and Chunk PDF**:
    ```python
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader("Kanski_Clinical_Ophthalmology.pdf")
    documents = loader.load_and_split()
    ```

5. **Create Vector Store**:
    ```python
    from langchain.vectorstores import Chroma
    vectorstore = Chroma.from_documents(documents, embeddings)
    ```

6. **Create Retriever**:
    ```python
    retriever = vectorstore.as_retriever()
    ```

7. **Define Prompt Template**:
    Includes `{context}`, `{question}`, `{chat_history}`, and instructions.

8. **Build RAG Chain**:
    ```python
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    ```

9. **Chat with the bot**:
    ```python
    response = qa_chain.run("What is the treatment for diabetic retinopathy?")
    ```

---

### 🔹 Phase 2: Add Conversational Memory

1. **History-Aware Retriever**:
    Integrates chat history to enhance document retrieval relevance.

2. **Conversational Retrieval Chain**:
    ```python
    from langchain.chains import ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    ```

3. **Manage Chat Sessions**:
    ```python
    chat_history = []
    query = "Tell me about age-related macular degeneration"
    result = chain.invoke({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    ```

---

## 💬 Example

```text
👤 User: What are the symptoms of glaucoma?
🤖 Bot: Common symptoms include eye pain, blurred vision, halos around lights, and vision loss...
