import streamlit as st
from model_codes.image_model import predict_image
from model_codes.multimodel_model import predict_multimodal
from model_codes.rag_model import rag_assistant
import time
import uuid

st.set_page_config(page_title="Eye Disease Assistant", page_icon="🧑‍⚕️", layout="wide")

# ---------------------------
# Session state initialization
# ---------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.chats["default"] = {"messages": [], "title": "Default Chat"}

if "active_chat" not in st.session_state:
    st.session_state.active_chat = "default"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "typing" not in st.session_state:
    st.session_state.typing = False

# Helper function to get active messages
def active_messages():
    return st.session_state.chats[st.session_state.active_chat]["messages"]

# Function to create a new chat
def new_chat():
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.chats[chat_id] = {"messages": [], "title": f"Chat {len(st.session_state.chats)}"}
    st.session_state.active_chat = chat_id
    st.session_state.messages = []
    return chat_id

# Function to delete a specific chat
def delete_chat(chat_id):
    if chat_id in st.session_state.chats and len(st.session_state.chats) > 1:
        del st.session_state.chats[chat_id]
        if st.session_state.active_chat == chat_id:
            st.session_state.active_chat = list(st.session_state.chats.keys())[0]
            st.session_state.messages = st.session_state.chats[st.session_state.active_chat]["messages"]

# ---------------------------
# Enhanced CSS Styling
# ---------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Title styling */
    .app-title {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #4a6fa5 0%, #2c3e50 100%);
        color: white;
        border-radius: 12px;
        font-size: 3.2rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        padding: 2.5rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        border-radius: 16px;
        border: 2px dashed #c3cfe2;
        margin: 2rem 0;
        color: #2d3748;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #4a6fa5 0%, #2c3e50 100%);
        color: white;
    }
    
    /* Chat container */
    .chat-container {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.05);
        max-height: 50vh;
        overflow-y: auto;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #7194EB 0%, #516EB5 100%);
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 12px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        line-height: 1.5;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #A2C4F2 0%, #C5DCF0 100%);
        color: #2d3748;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 12px 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        line-height: 1.5;
        border: 1px solid #e2e8f0;
    }
    
    /* Simple Input Area Styling */
    .input-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    
    /* Simple text area styling */
    .stTextArea > div > div > textarea {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 12px;
        font-size: 14px;
        line-height: 1.5;
        resize: vertical;
    }
    
    /* Simple file uploader styling */
    .stFileUploader > div {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px;
        background-color: #f9fafb;
    }
    
    .stFileUploader label {
        font-size: 14px;
        font-weight: 500;
        color: #374151;
    }
    
    /* Simple button styling */
    .stButton > button {
        background-color: #4a6fa5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #3d5a91;
    }
    
    .stButton > button:focus {
        outline: 2px solid #4a6fa5;
        outline-offset: 2px;
    }
    
    /* Chat title styling */
    .chat-title {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #4a6fa5 0%, #2c3e50 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 20px;
        background-color: #A2C4F2;
        border-radius: 18px;
        margin: 12px 0;
        max-width: 220px;
        animation: pulse 1.5s ease-in-out infinite alternate;
        color: black;
    }
    
    @keyframes pulse {
        from { opacity: 0.7; }
        to { opacity: 1; }
    }
    
    .typing-dots {
        display: inline-flex;
        gap: 6px;
    }
    
    .typing-dot {
        width: 10px;
        height: 10px;
        background-color: #3B4A6E;
        border-radius: 50%;
        animation: bounce 1.4s ease-in-out infinite both;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.8); }
        40% { transform: scale(1.2); }
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Sidebar button styling */
    .sidebar-button {
        width: 100%;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        text-align: center;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .sidebar-button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Enhanced Sidebar
# ---------------------------
with st.sidebar:
    st.markdown('<div class="chat-title">👁️ Eye Disease Assistant</div>', unsafe_allow_html=True)
    
    # New chat button (prominent)
    if st.button("🆕 New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
        new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Chat history with better UI
    st.subheader("💬 Chat History", anchor=False)
    
    if len(st.session_state.chats) > 0:
        for chat_id, chat_data in st.session_state.chats.items():
            is_active = chat_id == st.session_state.active_chat
            chat_title = chat_data.get("title", f"Chat {chat_id[:4]}")
            
            # Create columns for chat button and delete button
            col1, col2 = st.columns([4, 1])
            
            with col1:
                button_type = "primary" if is_active else "secondary"
                if st.button(
                    f"{'🔹 ' if is_active else '🔸 '}{chat_title}", 
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.active_chat = chat_id
                    st.session_state.messages = chat_data["messages"]
                    st.rerun()
            
            with col2:
                if len(st.session_state.chats) > 1:  # Don't allow deleting the last chat
                    if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                        delete_chat(chat_id)
                        st.rerun()
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics", anchor=False)
    active_chat_data = st.session_state.chats[st.session_state.active_chat]
    message_count = len(active_chat_data["messages"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Messages", message_count)
    with col2:
        st.metric("Total Chats", len(st.session_state.chats))
    
    st.markdown("---")
    
    # Information section
    st.subheader("ℹ️ About", anchor=False)
    st.info("This assistant helps with eye disease information and image analysis. For medical diagnosis, always consult a healthcare professional.")
    
    # Clear all button
    if st.button("🗑️ Clear All Chats", use_container_width=True, type="secondary"):
        if st.session_state.get('confirm_clear', False):
            st.session_state.chats = {}
            st.session_state.chats["default"] = {"messages": [], "title": "Default Chat"}
            st.session_state.active_chat = "default"
            st.session_state.messages = []
            st.session_state.confirm_clear = False
            st.rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("Click again to confirm clearing all chats")

# ---------------------------
# Main Chat Interface
# ---------------------------

# Title
st.markdown('<div class="app-title">👁️EyeDoc     |     Eye Disease Assistant</div>', unsafe_allow_html=True)

# Welcome message when no messages exist
if len(active_messages()) == 0:
    st.markdown('''
    <div class="welcome-message">
        <h3>👋 Hello! I'm EyeDoc ! Your Eye Disease Assistant</h3>
        <p>I can help you with information about eye diseases, analyze scanned eye images, and answer your questions.</p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-top: 24px;">
            <div style="background: rgba(74, 111, 165, 0.1); padding: 16px; border-radius: 12px;">
                <h4>💬 Ask Questions</h4>
                <p>Get information about eye conditions, symptoms, and treatments</p>
            </div>
            <div style="background: rgba(74, 111, 165, 0.1); padding: 16px; border-radius: 12px;">
                <h4>🖼️ Upload OCT Scan Images</h4>
                <p>Upload eye images for analysis and preliminary assessment</p>
            </div>
        </div>
        <p style="margin-top: 24px;"><b>Start by typing a message or uploading an image below!</b></p>
    </div>
    ''', unsafe_allow_html=True)

# Current chat title
current_chat_title = st.session_state.chats[st.session_state.active_chat]["title"]

# Chat messages container
with st.container():
    
    messages = active_messages()
    
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            st.markdown(f'''
            <div class="user-message">
                {f'<strong>You:</strong><br>{msg["content"]}' if msg.get("content") else '<strong>You uploaded an image:</strong>'}
            </div>
            ''', unsafe_allow_html=True)
            
            if msg.get("image"):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(msg["image"], caption="Your uploaded image", use_container_width=True)
        
        elif msg["role"] == "assistant":
            st.markdown(f'''
            <div class="assistant-message">
                <strong>Assistant:</strong><br>{msg["content"]}
            </div>
            ''', unsafe_allow_html=True)
    
    # Typing indicator
    if st.session_state.get('typing', False):
        st.markdown('''
        <div class="typing-indicator">
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            <span>EyeDoc is thinking</span>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
# Simple and Professional Input Area
# ---------------------------

# Use form to handle input clearing properly
with st.form(key="message_form", clear_on_submit=True):
    # Text input
    text_input = st.text_area(
        "Message",
        height=80,
        placeholder="Type your disease, symptoms, or concerns here...",
        key="message_text_input"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload OCT Images Only", 
        type=["png", "jpg", "jpeg"],
        key="image_file_input"
    )
    
    # Send button
    send_clicked = st.form_submit_button("Send", type="primary", use_container_width=True)

# Handle form submission
if send_clicked and (text_input.strip() or uploaded_file):
    query = text_input.strip()
    user_image = uploaded_file if uploaded_file else None

    # Add user message
    active_messages().append({
        "role": "user",
        "content": query if query else None,
        "image": user_image
    })

    # Set typing indicator
    st.session_state.typing = True
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Enhanced Message Processing
# ---------------------------
if st.session_state.get('typing', False):
    try:
        # Simulate processing time
        time.sleep(1.5)
        
        # Get the last user message
        last_message = active_messages()[-1]
        query = last_message.get("content", "")
        user_image = last_message.get("image")
        
        # Generate response based on input type
        if user_image and query:
            disease, confidence = predict_multimodal(user_image, query)
            answer = f"🔍 **Analysis Results:**\n\n**Predicted Condition:** {disease}\n**Confidence:** {confidence*100:.2f}%\n\n"
            
            # Get additional information
            qa_chain = rag_assistant.get_chain()
            extra = qa_chain.invoke({"question": f"Explain {disease} in simple terms and provide care recommendations."})["answer"]
            answer += f"**Detailed Information:**\n{extra}"

        elif user_image:
            disease, confidence = predict_image(user_image)
            answer = f"🔍 **Image Analysis Results:**\n\n**Predicted Condition:** {disease}\n**Confidence:** {confidence*100:.2f}%\n\n"
            
            qa_chain = rag_assistant.get_chain()
            extra = qa_chain.invoke({"question": f"Explain {disease} in simple terms and provide care recommendations."})["answer"]
            answer += f"**Detailed Information:**\n{extra}"

        elif query:
            qa_chain = rag_assistant.get_chain()
            answer = qa_chain.invoke({"question": query})["answer"]
        else:
            answer = "I'm sorry, I didn't receive any text or image to analyze. Please try again!"

        # Add AI response
        active_messages().append({"role": "assistant", "content": answer})
        
        # Update chat title if it's the first meaningful exchange
        if len(active_messages()) <= 2 and query:
            # Generate a title from the first user message
            title_words = query.split()[:4]  # First 4 words
            new_title = " ".join(title_words).title()
            if len(new_title) > 3:  # Only update if meaningful
                st.session_state.chats[st.session_state.active_chat]["title"] = new_title

    except Exception as e:
        # Add error message
        active_messages().append({
            "role": "assistant", 
            "content": f"⚠️ **Error Processing Request**\n\nI encountered an error while processing your request: {str(e)}\n\nPlease try again or contact support if the issue persists."
        })
    
    # Clear typing indicator
    st.session_state.typing = False
    st.rerun()

# ---------------------------
# Footer Information
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 0.5rem;">
    <small>
    👁️ <strong>Eye Disease Assistant</strong> • 
    Powered by AI • 
    For educational purposes only - consult a medical professional for diagnosis
    </small>
</div>
""", unsafe_allow_html=True)