from ultralytics import YOLO
import torch
import cv2 
import numpy as np
import pathlib
import matplotlib.pyplot as plt

model = YOLO('./YoloV8 Segmentation Model/runs/detect/weights/best.pt')


img_file='image3_1202_png.rf.44988022a22e7d398c4c02d9db60c285.jpg'
img = cv2.imread(f'./uploaded_images/{img_file}')
results = model(img)
#res_plotted = results[0].plot()

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    #print(masks)
    #print(type(masks))
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='./result_yolo/result.jpg')  # save to disk