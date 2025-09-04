import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import streamlit as st

# -----------------------------
# Fusion model definition
# -----------------------------
class FusionClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.img_proj = nn.Linear(2048, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768*2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        img_out = self.img_proj(image)
        combined = torch.cat((text_out, img_out), dim=1)
        return self.classifier(combined)

# -----------------------------
# Load models
# -----------------------------
@torch.no_grad()
def load_multimodal_model():
    # InceptionV3 for feature extraction
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
    output = GlobalAveragePooling2D()(base_model.output)
    inception_model = Model(inputs=base_model.input, outputs=output)

    # Fusion model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_model = FusionClassifier(num_classes=4).to(device)
    fusion_model.load_state_dict(torch.load("model_files/fusion_classifier.pth", map_location=device))
    fusion_model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return inception_model, fusion_model, tokenizer, device

inception_model, fusion_model, tokenizer, device = load_multimodal_model()

# -----------------------------
# Helper functions
# -----------------------------
def extract_image_feature(img_file):
    img = keras_image.load_img(img_file, target_size=(299,299))
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

    diagnosis_map = {0:"Glaucoma", 1:"Cataract", 2:"Diabetic Retinopathy", 3:"Normal"}
    return diagnosis_map[pred_class], confidence
