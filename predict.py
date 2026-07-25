import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model only once
model = tf.keras.models.load_model("model/brain_tumor_model.keras")

CLASS_NAMES = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

def predict_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    probabilities = {
        CLASS_NAMES[i]: float(prediction[i] * 100)
        for i in range(len(CLASS_NAMES))
    }

    return image, predicted_class, confidence, probabilities