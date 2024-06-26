# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/157k8ekLZOs7gNUqZ_0FebC1jOc9Xf4U_
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install ultralytics
import ultralytics
ultralytics.checks()

# Commented out IPython magic to ensure Python compatibility.
# %pip install zipfile
import zipfile
import os

def unzip_file(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))
        print("File successfully extracted.")
    except zipfile.BadZipFile as e:
        print(f"Error: {e}")

# Example usage:
zip_file_path = './archive.zip'
unzip_file(zip_file_path)

# Run inference on an image with YOLO
!yolo detect train data='./data.yaml' model=yolov8n-seg.pt epochs=90 imgsz=320 device=0