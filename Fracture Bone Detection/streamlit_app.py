import os
import requests
import streamlit as st
from PIL import Image



if __name__ == '__main__':
    st.title("Fracture Bone Detection")
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        with col1:
        #Display the image:
            st.image(uploaded_file, caption="Uploaded file", use_column_width = True)
        uploaded_folder = './uploaded_images'
        os.makedirs(uploaded_folder, exist_ok=True)
        uploaded_image_path = os.path.join(uploaded_folder, uploaded_file.name)

        with open(uploaded_image_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        files = {"img": open(uploaded_image_path,"rb")}

        fastapi_url = "http://127.0.0.1:8000/predict"
        response = requests.post(fastapi_url, files = files)
        
        with col2:
        #Display the prediction result
            if response.status_code == 200:
                result = response.json()
                st.write(f"Prediction: {result['result']}")
            else:
                st.write(f"Error predicting image. Status code: {response.status_code}")

