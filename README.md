# 🧑‍⚕️ Eye Disease Assistant

A multimodal AI assistant for **eye disease classification and explanation**, integrating **text-based symptom queries**, **OCT/eye image analysis**, and **multimodal fusion** with a **ChatGPT-style interface**.  

This project is part of a research initiative to improve diagnostic support using **vision transformers**, **RAG-based retrieval**, and **LLMs** for patient-friendly explanations.

---

## 🚀 Features

- **Multimodal Input**:  
  - Text-only (symptom descriptions) → uses **RAG + LLM** for answers.  
  - Image-only (eye/OCT images) → uses **CNN/InceptionV3-based classifier**.  
  - Multimodal (text + image) → uses **Fusion model (BERT + InceptionV3)**.

- **Interactive Chat Interface**:  
  - ChatGPT-style design with **sticky input bar** and **send button**.  
  - Inline **image uploader** for messages.  
  - Scrollable chat history with **auto-scroll** to latest message.  
  - **Typing indicator** while AI is generating responses.

- **Chat Management**:  
  - Start a **new chat** or **clear history** at any time.  
  - Preserves conversation history for context-aware answers.

- **Patient-Friendly Explanations**:  
  - Uses **RAG-based retrieval** to provide simple explanations of predicted eye diseases.  
  - Supports diseases: **Glaucoma, Cataract, Diabetic Retinopathy, Normal**.

---

## 🛠️ Architecture

```
    UserInput[User Input (Text / Image)] --> ChatInterface[Chat Interface]
    ChatInterface --> Controller
    Controller -->|Text Only| RAGModel[RAG + LLM]
    Controller -->|Image Only| ImageModel[CNN/InceptionV3 Classifier]
    Controller -->|Text + Image| MultiModel[Fusion Model: BERT + InceptionV3]
    MultiModel --> Response[AI Response]
    ImageModel --> Response
    RAGModel --> Response
    Response --> Gemini LLM
    Response --> User Interface
```

## 🧩 Components

- RAG Model	Conversational Retrieval-Augmented Generation for text-based queries using Google Gemini LLM.
- Image Model	CNN / InceptionV3-based classifier for OCT/eye images.
- Multimodal Model	Fusion model combining BERT embeddings and InceptionV3 image features for joint prediction.
- Frontend	Streamlit-based ChatGPT-style UI with inline image upload, sticky input bar, send button, typing indicator.
- Backend	Python logic handling model inference, RAG chain, and chat history management.

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/your-username/EyeDoc.git
```

2. Create a virtual environment:

```
python -m venv venv
venv\Scripts\activate     
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Set environment variables (e.g., GOOGLE_API_KEY for RAG model):
```
GOOGLE_API_KEY="your_api_key_here"
```

## 🖥️ Usage

- Run the Streamlit app:
```
streamlit run app.py
```

### Interface Features:

- Type symptoms or questions in the input box.
- Optionally upload an eye/OCT image for multimodal diagnosis.
- Click Send to generate predictions and explanations.
- Chat history scrolls above input; input bar remains sticky.
- Use New Chat or Clear History buttons to reset conversation.

## 📊 Supported Eye Diseases

- Glaucoma:	Eye condition leading to optic nerve damage.
- Cataract:	Clouding of the lens affecting vision.
- Diabetic Retinopathy:	Retinal damage due to diabetes.
- Normal:	No significant disease detected.

## 🔧 Research Notes

- Image Model: Pretrained InceptionV3 features + custom CNN classifier.
- Multimodal Model: BERT for text embeddings, InceptionV3 for image features, fused via fully connected layers.
- RAG Model: Retrieval-Augmented Generation using FAISS vectorstore + Google Gemini LLM.
- Chat History: Managed in st.session_state.messages for conversation continuity.

## 📁 Folder Structure

```
eye-disease-assistant/
│
├─ app.py                       # Streamlit front-end
├─ model_codes/
│   ├─ image_model.py           # CNN-based image classifier
│   ├─ multimodel_model.py      # Fusion multimodal model
│   └─ rag_model.py             # RAG + LLM chain
├─ vectorstore/                 # FAISS embeddings for RAG
├─ model_files/
|   |_ fusion_classifier.pth    # Saved multimodal model
|   ├─ my_model.keras           # Saved image model
└─ requirements.txt             # Python dependencies
```

## ⚡ Future Improvements

- Smooth, non-blinking chat UI using st.chat_message updates without rerun.
- Support multiple image uploads per message.
- Additional diseases & dataset expansion.
- Deployment on AWS / Azure for scalable usage.

## 📄 License

This project is for research purposes. Use responsibly and cite if used in publications.

## ✨ Acknowledgements

- LangChain
- HuggingFace Transformers
- Streamlit
- Kaggle OCT & Eye datasets
- Kanski’s Clinical Ophthalmology (text resources)

