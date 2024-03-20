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


#import model
import resnet_model
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv

app = FastAPI()

#device = 'cuda:0' if t.cuda.is_available() else 'cpu'

device = 'cpu'
resnet_model = resnet_model.ResNet()
#resnet_model = model3.ResNetWithSegmentation(num_classes=7)
resnet_model.to(device)
ckp = t.load('./Resnet Model/resnet_checkpoint_131.ckp', device)
resnet_model.load_state_dict(ckp['state_dict'])
resnet_model.eval()
fracture_names= ['Elbow positive', 'Fingers positive', 
            'Forearm fracture', 'Humerus fracture', 
            'Humerus fracture', 'Shoulder fracture', 'Wrist positive']
transformation = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((512, 512)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

""" def prediction(image_path):
    image = imread(image_path)
    if len(image.shape) == 2:
        image = gray2rgb(image)
    x = transformation(image)
    x = x.resize_(1,3,512, 512)
    x = x.to(device)
    with t.no_grad():
        output = resnet_model(x)
    output = output.detach().cpu().numpy()[0]
    output = output/np.sum(output)
    label = fracture_names[np.argmax(output)]
    return label """

def prediction(image_path):
    image = imread(image_path)
    if len(image.shape) == 2:
        image = gray2rgb(image)
    x = transformation(image)
    x = x.resize_(1,3,512, 512)
    x = x.to(device)
    with t.no_grad():
        output, segmentation_output = resnet_model(x)
    segmentation_map = segmentation_output.detach().cpu().numpy()[0]
    output = output.detach().cpu().numpy()[0]
    output = output/np.sum(output)
    label = fracture_names[np.argmax(output)]
    segmentation_map = segmentation_map[np.argmax(output)]
    return label, segmentation_map, image

@app.post('/predict')
async def predict_endpoint(img: UploadFile = (...)):
    img_path = os.path.join(os.getcwd(), 'uploaded_images', img.filename)
    label, segmentation_map, input_image_np = prediction(img_path)

    height = input_image_np.shape[0]
    width = input_image_np.shape[1]
    segmentation_map_resized = resize(segmentation_map, 
                                    (height, width), 
                                    mode='constant', anti_aliasing=True)
    threshold = 0.5  # You can adjust the threshold as needed
    binary_mask = segmentation_map_resized > threshold
    input_image_rgba = np.concatenate([input_image_np, np.ones((*input_image_np.shape[:2], 1), dtype=np.uint8) * 255], axis=-1)
    # Create a mask for the segmented regions
    segmented_overlay = np.zeros_like(input_image_rgba)
    segmented_overlay[binary_mask] = [10, 255, 10, 200]  # Set segmented regions to red with full opacity

    # Overlay the segmented regions on the original image
    overlay_image = np.maximum(input_image_rgba, segmented_overlay)
    # Convert the NumPy array (segmented_overlay) to a PIL Image
    overlay_image_pil = Image.fromarray(overlay_image)
    # Create a buffer to hold the image data
    buffered = io.BytesIO()
    # Save the PIL Image to the buffer in PNG format
    overlay_image_pil.save(buffered, format="PNG")
    # Convert the buffered image data to base64
    overlay_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return JSONResponse(content={"result": label, "segmented": overlay_image_base64})


if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)