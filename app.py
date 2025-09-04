import streamlit as st
from model_codes.image_model import predict_image
from model_codes.multimodel_model import predict_multimodal
from model_codes.rag_model import rag_assistant
import time

st.set_page_config(page_title="Eye Disease Assistant", page_icon="🧑‍⚕️", layout="wide")
st.title("🧑‍⚕️ Eye Disease Assistant")

# -----------------------------
# Session state
# -----------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("temp_image", None)
st.session_state.setdefault("input_text", "")

# -----------------------------
# Chat control buttons
# -----------------------------
col1, col2 = st.columns([1,1])
if col1.button("🆕 New Chat"):
    st.session_state.messages = []
    st.session_state.temp_image = None
    st.session_state.input_text = ""
if col2.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.temp_image = None
    st.session_state.input_text = ""

# -----------------------------
# Display chat messages
# -----------------------------
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("content"):
                st.markdown(msg["content"])
            if msg.get("image"):
                st.image(msg["image"], caption="Uploaded Image", use_container_width=True)

# -----------------------------
# Sticky chat input at bottom
# -----------------------------
input_container = st.container()
with input_container:
    cols = st.columns([8, 1])
    # Text input
    st.session_state.input_text = cols[0].text_input("Type your message...", value=st.session_state.input_text, key="chat_input")
    # Image uploader inside chat bar
    uploaded_file = cols[0].file_uploader("Optional: Upload image", type=["png","jpg","jpeg"], key="file_upload", label_visibility="collapsed")
    send_button = cols[1].button("Send")

# -----------------------------
# Handle sending message
# -----------------------------
if send_button:
    query = st.session_state.input_text.strip()
    user_image = uploaded_file if uploaded_file else None

    # Reset input fields
    st.session_state.input_text = ""
    st.session_state.temp_image = None

    # Append user message
    if query or user_image:
        st.session_state.messages.append({
            "role": "user",
            "content": query if query else None,
            "image": user_image
        })

        # Display user message immediately
        with st.chat_message("user"):
            if query:
                st.markdown(query)
            if user_image:
                st.image(user_image, caption="Uploaded Image", use_container_width=True)

        # -----------------------------
        # Typing indicator
        # -----------------------------
        placeholder = st.empty()
        with placeholder.container():
            st.chat_message("assistant").markdown("Typing... ⏳")

        # -----------------------------
        # Generate AI response
        # -----------------------------
        try:
            time.sleep(1)  # simulate typing

            if user_image and query:
                disease, confidence = predict_multimodal(user_image, query)
                answer = f"Prediction: **{disease}** ({confidence*100:.2f}%)"
                qa_chain = rag_assistant.get_chain()
                extra = qa_chain.invoke({"question": f"Explain {disease} in simple terms."})["answer"]
                answer += "\n\n" + extra

            elif user_image:
                disease, confidence = predict_image(user_image)
                answer = f"Prediction: **{disease}** ({confidence*100:.2f}%)"
                qa_chain = rag_assistant.get_chain()
                extra = qa_chain.invoke({"question": f"Explain {disease} in simple terms."})["answer"]
                answer += "\n\n" + extra

            elif query:
                qa_chain = rag_assistant.get_chain()
                answer = qa_chain.invoke({"question": query})["answer"]

            # Remove typing indicator
            placeholder.empty()

            # Append AI response
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(answer)

        except Exception as e:
            placeholder.empty()
            st.error(f"❌ Error: {e}")
