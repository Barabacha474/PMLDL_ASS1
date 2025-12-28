import streamlit as st
import requests
import base64

st.title("Image Classifier App")

# api_url = "http://api:8000/predict"
api_url = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    data = {'image': encoded_image}
    response = requests.post(api_url, json=data)

    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.success(f"Predicted Class: {prediction}")
    else:
        st.error(f"Error: {response.text}")