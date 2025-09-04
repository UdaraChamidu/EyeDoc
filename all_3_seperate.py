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
from langchain_google_genai import ChatGoogleGenerativeAI

import tensorflow as tf
import numpy as np
from PIL import Image

# Torch / Transformers for multimodal model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# -----------------------------
# Load CNN (Image-Only Model)
# -----------------------------
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("model_files/my_model.keras")

cnn_model = load_cnn_model()

def preprocess_image(image_data, target_size=(256, 256)):
    img = Image.open(image_data).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# -----------------------------
# Load Multimodal Model (Fusion)
# -----------------------------
class FusionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FusionClassifier, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.img_proj = nn.Linear(2048, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        img_out = self.img_proj(image)
        combined = torch.cat((text_out, img_out), dim=1)
        return self.classifier(combined)

@st.cache_resource
def load_multimodal_model():
    # InceptionV3 feature extractor
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    output = GlobalAveragePooling2D()(base_model.output)
    inception_model = Model(inputs=base_model.input, outputs=output)

    # FusionClassifier
    num_classes = 4  # Cataract, Diabetic Retinopathy, Glaucoma, Normal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionClassifier(num_classes).to(device)
    model.load_state_dict(torch.load("model_files/fusion_classifier.pth", map_location=device))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return inception_model, model, tokenizer, device

inception_model, fusion_model, tokenizer, device = load_multimodal_model()

def extract_image_feature(img_file):
    img = keras_image.load_img(img_file, target_size=(299, 299))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = inception_model.predict(x)
    return features[0]

def predict_multimodal(img_file, caption_text):
    img_feat = extract_image_feature(img_file)
    img_tensor = torch.tensor(img_feat, dtype=torch.float32).unsqueeze(0).to(device)

    tokens = tokenizer(caption_text, padding='max_length', truncation=True, max_length=50, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        logits = fusion_model(img_tensor, input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    diagnosis_map = {0: "Glaucoma", 1: "Cataract", 2: "Diabetic Retinopathy", 3: "Normal"}
    return diagnosis_map[pred_class], confidence

# -----------------------------
# RAG Setup
# -----------------------------
load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="Eye Disease RAG Assistant", page_icon="🧑‍⚕️")
st.title("🧑‍⚕️ Eye Disease Assistant")

st.session_state.setdefault("mode", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("memory", None)

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
        st.success("✅ Ready to assist with eye diseases!")
    except Exception as e:
        st.error(f"❌ Failed to load vectorstore: {e}")

if st.session_state.memory is None and st.session_state.vectorstore:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt_template = PromptTemplate.from_template("""
You are an AI assistant specialized in Eye Diseases.
Use the following context to answer.
If question is unclear, ask for more info.
Keep answers friendly, simple, and accurate.

Context:
{context}

Question: {question}
Answer:
""")

def get_qa_chain():
    llm = ChatGoogleGenerativeAI(
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",
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

# -----------------------------
# Mode Selection
# -----------------------------
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

# -----------------------------
# 🖼️ Image-Only Mode
# -----------------------------
if st.session_state.mode == "image":
    st.subheader("🖼️ Image Input Mode")
    uploaded_file = st.file_uploader("Upload an Eye Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Eye Image", use_container_width=True)
        img_array = preprocess_image(uploaded_file)
        prediction = cnn_model.predict(img_array)

        class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
        predicted_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_index]
        predicted_disease = class_names[predicted_index]

        st.success(f"Prediction: **{predicted_disease}** with confidence {confidence*100:.2f}%")

        qa_chain = get_qa_chain()
        overview = qa_chain.invoke({"question": f"Explain {predicted_disease} in patient-friendly terms."})
        st.markdown(overview["answer"])

# -----------------------------
# 💬 Text-Only Mode
# -----------------------------
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
            st.chat_message("assistant").markdown(result["answer"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        except Exception as e:
            st.error(f"❌ Error: {e}")

# -----------------------------
# 🔀 Multimodal Mode
# -----------------------------
elif st.session_state.mode == "both":
    st.subheader("🔀 Multimodal Image + Text Input Mode")
    uploaded_file = st.file_uploader("Upload your Eye Image", type=["png", "jpg", "jpeg"])
    query = st.chat_input("Add text symptoms or questions here...")

    if uploaded_file and query:
        st.image(uploaded_file, caption="Uploaded OCT Image", use_container_width=True)

        with st.spinner("Analyzing image and text together..."):
            disease, confidence = predict_multimodal(uploaded_file, query)
            st.success(f"Multimodal Prediction: **{disease}** with confidence {confidence*100:.2f}%")

            qa_chain = get_qa_chain()
            overview = qa_chain.invoke({"question": f"Explain {disease} in patient-friendly terms."})
            st.markdown(overview["answer"])
