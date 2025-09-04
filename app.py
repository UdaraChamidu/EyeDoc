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

# -----------------------------
# Chat control buttons
# -----------------------------
col1, col2 = st.columns([1,1])
if col1.button("🆕 New Chat"):
    st.session_state.messages = []
if col2.button("🗑️ Clear Chat History"):
    st.session_state.messages = []

# -----------------------------
# CSS for sticky input
# -----------------------------
st.markdown("""
<style>
.chat-scroll {
    max-height: 70vh;
    overflow-y: auto;
    padding-bottom: 120px; 
}
.chat-input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: white;
    padding: 10px 20px;
    border-top: 1px solid #ddd;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Chat messages display
# -----------------------------
st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], caption="Uploaded Image", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Sticky chat input using form
# -----------------------------
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([8,1])
    text_input = cols[0].text_input("Type your message...", label_visibility="collapsed")
    uploaded_file = cols[0].file_uploader("Upload image (optional)", type=["png","jpg","jpeg"], label_visibility="collapsed")
    send_button = cols[1].form_submit_button("Send")  # Pressing Enter triggers this

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Handle sending
# -----------------------------
if send_button and (text_input.strip() or uploaded_file):
    query = text_input.strip()
    user_image = uploaded_file if uploaded_file else None

    # Append user message
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

    # Typing indicator
    placeholder = st.empty()
    with placeholder.container():
        st.chat_message("assistant").markdown("Typing... ⏳")

    # -----------------------------
    # Generate AI response
    # -----------------------------
    try:
        time.sleep(1)

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

        placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(answer)

    except Exception as e:
        placeholder.empty()
        st.error(f"❌ Error: {e}")


