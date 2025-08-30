import os
import logging
import warnings
import streamlit as st
from dotenv import load_dotenv
  
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ Gemini API

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
from io import StringIO

# Load your CNN model once
@st.cache_resource  # cache so it loads only once
def load_cnn_model():
    model = tf.keras.models.load_model("my_model.keras")
    return model
    
cnn_model = load_cnn_model()

def preprocess_image(image_data, target_size=(256, 256)):
    img = Image.open(image_data).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)  # <-- no /255.0 here
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array


# Load environment variables
load_dotenv()

# Suppress unnecessary warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="Eye Disease RAG Assistant", page_icon="🧑‍⚕️")
st.title("🧑‍⚕️ Eye Disease Assistant")

# Initialize session state variables
st.session_state.setdefault("mode", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("memory", None)

# Load vectorstore
if st.session_state.vectorstore is None:
    try:
        st.info("🔄 Loading ... Please wait !")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
        vectorstore = FAISS.load_local(
            "vectorstore/eye_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vectorstore = vectorstore
        st.success("✅ Hello there! I am ready to assist you with eye diseases. Ask me anything related to eye health.")
    except Exception as e:
        st.error(f"❌ Failed to load vectorstore: {e}")

# Load memory
if st.session_state.memory is None and st.session_state.vectorstore:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# cnn prediction send to RAG
def summarize_predicted_disease(disease_name: str, confidence: float) -> str:
    # Build the question we’ll send to RAG/LLM
    summary_prompt = f"""
A CNN classified an uploaded eye image as: {disease_name} with confidence {confidence:.0%}.

Write a concise, patient-friendly overview with bullet points:
- What it is (1–2 sentences)
- Common symptoms
- How it’s diagnosed (key exams/tests)
- Evidence-based treatments (first-line and advanced)
- When it’s urgent / red flags
- Prevention and risk factors

Keep it ~150–200 words. Avoid jargon. If uncertain, mention that a clinical exam is needed.
"""

    # Prefer RAG if the vectorstore is available, else fall back to plain LLM
    try:
        if st.session_state.get("vectorstore") is not None:
            qa_chain = get_qa_chain()
            result = qa_chain.invoke({"question": summary_prompt})
            return result["answer"]
        else:
            llm = ChatGoogleGenerativeAI(
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                model="gemini-2.5-flash",
                temperature=0.2
            )
            msg = llm.invoke(summary_prompt)
            return msg.content
    except Exception as e:
        st.warning(f"RAG/LLM call failed: {e}")
        return "Unable to generate a summary right now. Please try again."

def summarize_differential(confidence: float) -> str:
    diff_prompt = f"""
The model's confidence is only {confidence:.0%}. Provide a brief caution plus a short differential overview among Cataract, Diabetic Retinopathy, and Glaucoma:
- Key distinguishing symptoms/signs
- Typical tests to confirm
- What the patient should do next

Limit to ~120 words, bullet points, patient-friendly.
"""
    try:
        if st.session_state.get("vectorstore") is not None:
            qa_chain = get_qa_chain()
            result = qa_chain.invoke({"question": diff_prompt})
            return result["answer"]
        else:
            llm = ChatGoogleGenerativeAI(
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                model="gemini-2.5-flash",
                temperature=0.2
            )
            msg = llm.invoke(diff_prompt)
            return msg.content
    except Exception:
        return ""



# Mode selection
if st.session_state.mode is not None:
    if st.button("🔙 Back to Mode Selection"):
        st.session_state.mode = None
        st.session_state.messages = []
        st.rerun()

if st.session_state.mode is None:
    st.subheader("Choose Input Mode")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🖼️ Image Input"):
            st.session_state.mode = "image"
            st.rerun()
    with col2:
        if st.button("💬 Text Input"):
            st.session_state.mode = "text"
            st.rerun()
    with col3:
        if st.button("🔀 Image + Text Input"):
            st.session_state.mode = "both"
            st.rerun()
    st.stop()

# Shared prompt
prompt_template = PromptTemplate.from_template("""
- You are an AI assistant specialized in Eye Diseases.

Tasks:
- Answer the question using only the following context.
- If the question is not about eye diseases, politely decline.
- Use the context to give a helpful, clear and explainable answer.
- If the question is unclear, ask the user for more info.
- Give answers in a friendly and easy-to-read way.
- if the provided content have not the correct answers, answer for the query according to your knowledge

Context:
{context}

Question: {question}
Answer:
""")

# Shared function to create QA chain
def get_qa_chain():
    llm = ChatGoogleGenerativeAI(
        google_api_key=os.environ.get("GOOGLE_API_KEY"),  # ✅ Gemini API key
        model="gemini-2.5-flash",  # ✅ Your model
        temperature=0
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=False,
        output_key="answer"
    )

# 🖼️ Image-Only Mode
if st.session_state.mode == "image":
    st.subheader("🖼️ Image Input Mode")
    uploaded_file = st.file_uploader("Upload an Eye Image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Eye Image", use_container_width=True)
        
        # Preprocess image for CNN
        img_array = preprocess_image(uploaded_file)

        # Predict
        prediction = cnn_model.predict(img_array)
        st.write("Raw prediction:", prediction)

        class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
        predicted_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_index]
        predicted_disease = class_names[predicted_index]

        st.success(f"Prediction: **{predicted_disease}** with confidence {confidence*100:.2f}%")

        # NEW: Ask RAG/LLM for a short overview and treatments
        with st.spinner("Generating information about the predicted disease..."):
            overview = summarize_predicted_disease(predicted_disease, confidence)
            st.markdown(overview)

        # Optional: If the model isn’t confident, show a brief differential
        if confidence < 0.60:
            with st.expander("Model is unsure — see possible alternatives"):
                diff = summarize_differential(confidence)
                if diff:
                    st.markdown(diff)
  
  
# 💬 Text-Only Mode
elif st.session_state.mode == "text":
    st.subheader("💬 Text Input Mode")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Describe your symptoms or ask a question...")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        try:
            qa_chain = get_qa_chain()
            result = qa_chain.invoke({"question": query})
            answer = result["answer"]
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"❌ Error during query: {e}")

# 🔀 Combined Mode
elif st.session_state.mode == "both":
    st.subheader("🔀 Combined Image + Text Input Mode")
    uploaded_file = st.file_uploader("Upload your Eye Image", type=["png", "jpg", "jpeg"])
    query = st.chat_input("Add text symptoms or questions here...")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded OCT Image", use_container_width=True)

    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        try:
            qa_chain = get_qa_chain()
            result = qa_chain.invoke({"question": query})
            answer = result["answer"]
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"❌ Error during query: {e}")
