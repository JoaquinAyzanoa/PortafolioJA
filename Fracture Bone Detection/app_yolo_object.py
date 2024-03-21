import os
import json
import uvicorn
import numpy as np
import torch as t
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from skimage.transform import resize
from PIL import Image
import io
import base64


#import resnet model
import resnet_model
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv

#import YOLO model
from ultralytics import YOLO
YOLOmodel = YOLO('./YoloV8 Object Model/runs/detect/weights/best.pt')

app = FastAPI()



def prediction(image_path):
    image = imread(image_path)
    YOLOresults = YOLOmodel(image)
    # Process results list
    for result in YOLOresults:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        #result.show()  # display to screen
        result.save(filename='./result_yolo/result.jpg')  # save to disk
    
    label = 'YoloV8: Object Detection '
    return label , YOLOresults


@app.post('/predict')
async def predict_endpoint(img: UploadFile = (...)):
    img_path = os.path.join(os.getcwd(), 'uploaded_images', img.filename)
    label, YOLOresults = prediction(img_path)

    image = imread('./result_yolo/result.jpg')
    # Convert the NumPy array (segmented_overlay) to a PIL Image
    overlay_image_pil = Image.fromarray(image)
    # Create a buffer to hold the image data
    buffered = io.BytesIO()
    # Save the PIL Image to the buffer in PNG format
    overlay_image_pil.save(buffered, format="PNG")
    # Convert the buffered image data to base64
    overlay_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return JSONResponse(content={"result": label, "segmented": overlay_image_base64})


if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)