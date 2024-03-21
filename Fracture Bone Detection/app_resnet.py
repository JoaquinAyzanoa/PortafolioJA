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



def prediction(image_path):
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

    return label , image

@app.post('/predict')
async def predict_endpoint(img: UploadFile = (...)):
    img_path = os.path.join(os.getcwd(), 'uploaded_images', img.filename)
    label , image = prediction(img_path)

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