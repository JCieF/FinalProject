import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

def preprocess_image(img):
    img = img.resize((128, 128)) 
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

model = load_model('finalExamIMG.keras')
classes = ['Amoeba', 'Euglena', 'Hydra', 'Paramecium', 'Rod_bacteria', 'Spherical bacteria', 'Spiral bacteria', 'Yeast']

st.title("Bacteria Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    array = preprocess_image(img)
    prediction = model.predict(array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = classes[predicted_class_index]

    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.markdown(f"<h2 style='font-size: 24px;'>Bacteria Type: {predicted_class_name}</h2>", unsafe_allow_html=True)
