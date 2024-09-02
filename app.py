import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from PIL import Image
import matplotlib.pyplot as plt
import os

# Define the gender dictionary with reversed values
gender_dict = {0: 'Male', 1: 'Female'}

# Define or import custom objects if needed
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

custom_objects = {
    'mae': mae
}

# Define the path to your model file
model_path = 'model.h5'

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file not found at: {model_path}")
        return None

model = load_model()

if model is None:
    st.stop()

st.title("Gender and Age Predictor")
st.write("Upload an image to predict the gender and age.")

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(128, 128)):
    # Load and preprocess the image
    img = load_img(image_path, color_mode='grayscale')
    img = img.resize(target_size, Image.Resampling.LANCZOS)  # Updated to use LANCZOS
    img = np.array(img)
    img = img / 255.0  # Normalize if necessary
    img = img.reshape(1, target_size[0], target_size[1], 1)  # Reshape for model input
    return img

# Image uploader
uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    new_image = preprocess_image(uploaded_file)
    
    # Perform prediction
    try:
        pred = model.predict(new_image)
        pred_gender = gender_dict[round(float(pred[0][0]))]  # Use updated gender_dict
        pred_age = round(float(pred[1][0]))

        # Display results
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write(f"Predicted Gender: {pred_gender}")
        st.write(f"Predicted Age: {pred_age} years")

        # Optionally, display the image
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(new_image.reshape(128, 128), cmap='gray')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

