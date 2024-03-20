from ultralytics import YOLO
import torch
import cv2 
import numpy as np
import pathlib
import matplotlib.pyplot as plt

model = YOLO('./YoloV8/runs/detect/weights/best.pt')


img_file='image1_3_png.rf.4f3936b1954ddb019efef8efe3594f6e.jpg'
img = cv2.imread(f'./data/train/images/{img_file}')
results = model(img)
#res_plotted = results[0].plot()

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk