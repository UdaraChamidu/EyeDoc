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
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Title styling */
    .app-title {
        text-align: top;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-size: 2rem;
        font-weight: bold;
        
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        padding: 2rem;
        background-color: #f8f9ff;
        border-radius: 15px;
        border: 2px dashed #667eea;
        margin: 2rem 0;
        color: #2c3e50;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Chat container */
    .chat-container {
        height: 0vh;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Input area styling */
    .input-container {
        
    }
    
    /* Chat title styling */
    .chat-title {
        text-align: center;
        color: #333;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 8px;
        border: 2px dashed #667eea;
        padding: 1rem;
        text-align: center;
        background-color: #f8f9ff;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 16px;
        background-color: #e9ecef;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 200px;
        animation: pulse 1.5s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from { opacity: 0.6; }
        to { opacity: 1; }
    }
    
    .typing-dots {
        display: inline-flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: #667eea;
        border-radius: 50%;
        animation: bounce 1.4s ease-in-out infinite both;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        color: #666;
        font-style: normal;
        padding: 2rem;
        background-color: #f8f9ff;
        border-radius: 10px;
        border: 2px dashed #667eea;
        margin: 2rem 0;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Enhanced Sidebar
# ---------------------------
with st.sidebar:
    st.markdown('<div class="chat-title">🧑‍⚕️ Eye Disease Assistant</div>', unsafe_allow_html=True)
    
    # New chat button (prominent)
    if st.button("✨ Start New Chat", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Chat history with better UI
    st.subheader("📋 Chat History")
    
    if len(st.session_state.chats) > 0:
        for chat_id, chat_data in st.session_state.chats.items():
            is_active = chat_id == st.session_state.active_chat
            chat_title = chat_data.get("title", f"Chat {chat_id[:4]}")
            
            # Create columns for chat button and delete button
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if st.button(
                    f"{'🟢 ' if is_active else '⚪ '}{chat_title}", 
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
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
    active_chat_data = st.session_state.chats[st.session_state.active_chat]
    message_count = len(active_chat_data["messages"])
    st.metric("Messages in current chat", message_count)
    st.metric("Total chats", len(st.session_state.chats))
    
    st.markdown("---")
    
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
st.markdown('<div class="app-title">🧑‍⚕️ Eye Disease Assistant</div>', unsafe_allow_html=True)

# Welcome message when no messages exist
if len(active_messages()) == 0:
    st.markdown('''
    <div class="welcome-message">
        <h3>👋 Hello! I'm your Eye Disease Assistant</h3>
        <p>I can help you with information about eye diseases, analyze eye images, and answer your questions.</p>
        <p>You can:</p>
        <ul style="text-align: left;">
            <li>Ask me about eye conditions and symptoms</li>
            <li>Upload an image of an eye for analysis</li>
            <li>Get information about treatments and prevention</li>
        </ul>
        <b>Start by typing a message or uploading an image below!</b>
    </div>
    ''', unsafe_allow_html=True)

# Current chat title
current_chat_title = st.session_state.chats[st.session_state.active_chat]["title"]
st.markdown(f'<div class="chat-title">💬 {current_chat_title}</div>', unsafe_allow_html=True)

# Chat messages container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
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
            <span>Assistant is thinking</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------
# Enhanced Input Area
# ---------------------------
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Create input form
with st.form(key="chat_form", clear_on_submit=True):
    # Text input
    text_input = st.text_area(
        "Type your message here...", 
        height=100, 
        placeholder="Ask me about eye diseases, symptoms, or upload an image for analysis...",
        label_visibility="collapsed"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "📎 Upload an eye image (optional)", 
        type=["png", "jpg", "jpeg"],
        help="Upload an image of an eye for disease detection"
    )
    
    # Submit button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        send_button = st.form_submit_button("🚀 Send Message", use_container_width=True, type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Enhanced Message Processing
# ---------------------------
if send_button and (text_input.strip() or uploaded_file):
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

# Process AI response if typing
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
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>
    🧑‍⚕️ <strong>Eye Disease Assistant</strong> • 
    Powered by AI • 
    For educational purposes only - consult a medical professional for diagnosis
    </small>
</div>
""", unsafe_allow_html=True)