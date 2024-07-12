import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

model = load_model('finalExamIMG.keras')

def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

st.title("Image Classification with My Model")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    array = preprocess_image(img)

    prediction = model.predict(array)
    predicted_class = np.argmax(prediction, axis=1)

    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted class: {predicted_class[0]}')

    st.write('Prediction probabilities:', prediction)
