from PIL import Image
import numpy as np

def preprocess_image(image_data, target_size=(256,256)):
    img = Image.open(image_data).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array
