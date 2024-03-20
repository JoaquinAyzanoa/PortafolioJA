#streamlit run streamlit_app.py
import os
import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import base64
import io
import numpy as np

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
                #st.write(f"Segmented image: {result['segmented']}")

                segmented_overlay_base64 = result.get('segmented', None)
                if segmented_overlay_base64:
                    # Convert base64 to bytes
                    segmented_overlay_bytes = base64.b64decode(segmented_overlay_base64)
                    # Convert bytes to numpy array
                    segmented_overlay_array = np.array(Image.open(io.BytesIO(segmented_overlay_bytes)))
                    # Plot the segmented overlay image
                    fig, ax = plt.subplots()
                    ax.imshow(segmented_overlay_array)
                    ax.axis('off')
                    st.pyplot(fig)

            else:
                st.write(f"Error predicting image. Status code: {response.status_code}")

