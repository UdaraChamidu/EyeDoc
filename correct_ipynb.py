import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

img_path = '/content/drive/MyDrive/old_state_of_art/dataset/diabetic_retinopathy/1000_right.jpeg'

# Load and resize
img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension and normalize
img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0

# Predict
pred = model.predict(img_array)
predicted_class = np.argmax(pred[0])
confidence = np.max(pred[0])

print(f"Predicted class: {class_names[predicted_class]} ({confidence*100:.2f}%)")

# Plot image with title
plt.imshow(img)
plt.title(f"Prediction: {class_names[predicted_class]}")
plt.axis('off')
plt.show()
