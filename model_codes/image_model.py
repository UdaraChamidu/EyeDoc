import tensorflow as tf
import numpy as np
from PIL import Image
from utils.helpers import preprocess_image
import streamlit as st

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("model_files/my_model.keras")

cnn_model = load_cnn_model()

def predict_image(uploaded_file):
    img_array = preprocess_image(uploaded_file)
    prediction = cnn_model.predict(img_array)
    class_names = ['Cataract','Diabetic Retinopathy','Glaucoma','Normal']
    pred_idx = np.argmax(prediction[0])
    disease, confidence = class_names[pred_idx], prediction[0][pred_idx]
    return disease, confidence
