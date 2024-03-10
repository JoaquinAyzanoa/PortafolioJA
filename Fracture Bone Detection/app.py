import os
import json
import uvicorn
import numpy as np
import torch as t
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import model
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv

app = FastAPI()

#device = 'cuda:0' if t.cuda.is_available() else 'cpu'

device = 'cpu'
resnet_model = model.ResNet()
resnet_model.to(device)
ckp = t.load('./checkpoints/checkpoint_131.ckp', device)
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
    return label

@app.post('/predict')
async def predict_endpoint(img: UploadFile = (...)):
    img_path = os.path.join(os.getcwd(), 'uploaded_images', img.filename)
    y = prediction(img_path)
    return JSONResponse(content={"result": y})


if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)