import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Load the model
model = load_model('finalExamIMG.keras')

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

# Streamlit app
st.title("Image Classification with My Model")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    array = preprocess_image(img)

    # Predict the class
    prediction = model.predict(array)
    predicted_class = np.argmax(prediction, axis=1)

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted class: {predicted_class[0]}')

    # Optionally, show the prediction probabilities
    st.write('Prediction probabilities:', prediction)
