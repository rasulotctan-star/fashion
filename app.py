
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion.keras")

model = load_model()

# Fashion MNIST class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Streamlit app
st.title("Fashion MNIST Clothing Classifier")
st.write("Upload a clothing image to classify it.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=250)

    # Preprocess image
    image = image.resize((28, 28))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    if st.button("Predict"):
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100

        st.success(f"Prediction: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}%")