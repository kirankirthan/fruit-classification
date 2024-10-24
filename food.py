import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('food_classification_model.h5')

# Function to preprocess the image (resize to the required size)
def preprocess_image(img):
    img = cv2.resize(np.array(img), (150, 150))  # Assuming your model was trained on 150x150 images
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image (same as training)
    return img

# Set up the Streamlit interface
st.title("Apple Quality Detection")
st.write("Take a photo to detect if the fruit is good or rotten.")

# Use the webcam to take a picture
picture = st.camera_input("Take a picture")

# When the user takes a picture
if picture:
    # Load the image taken by the user
    img = Image.open(picture)

    # Display the uploaded picture
    st.image(img, caption="Your fruit", use_column_width=True)

    # Preprocess the image for the model
    processed_image = preprocess_image(img)

    # Perform prediction
    prediction = model.predict(processed_image)

    # Interpret the result
    if prediction >= 0.5:
        st.write("This is a *Good fruit* !")
    else:
        st.write("This is a *Rotten fruit* .")