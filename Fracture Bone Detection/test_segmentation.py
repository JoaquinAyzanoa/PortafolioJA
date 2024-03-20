import os
import json
import uvicorn
import numpy as np
import torch as t
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean

import model3
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv

#device = 'cuda:0' if t.cuda.is_available() else 'cpu'

device = 'cpu'
resnet_model = model3.ResNetWithSegmentation(num_classes=7)
resnet_model.to(device)
ckp = t.load('./checkpoints/checkpoint_132.ckp', device)
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
        output, segmentation_output = resnet_model(x)
    segmentation_map = segmentation_output.detach().cpu().numpy()[0]
    output = output.detach().cpu().numpy()[0]
    output = output/np.sum(output)
    label = fracture_names[np.argmax(output)]
    segmentation_map = segmentation_map[np.argmax(output)]
    return label, segmentation_map, image

imagen_path = './uploaded_images/distal-humeral-fractures-2-_JPEG.rf.525ce876785d0fd798ec3af1593e5bc1.jpg'

label, segmentation_map, input_image_np = prediction(imagen_path)
print('Prediction: ', label)

num_classes = 7
colors = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow']



height = input_image_np.shape[0]
width = input_image_np.shape[1]
segmentation_map_resized = resize(segmentation_map, 
                                 (height, width), 
                                 mode='constant', anti_aliasing=True)
threshold = 0.5  # You can adjust the threshold as needed
binary_mask = segmentation_map_resized > threshold
image_segmented = binary_mask.squeeze()

input_image_rgba = np.concatenate([input_image_np, np.ones((*input_image_np.shape[:2], 1), dtype=np.uint8) * 255], axis=-1)
# Create a mask for the segmented regions
segmented_overlay = np.zeros_like(input_image_rgba)
segmented_overlay[binary_mask] = [10, 255, 10, 200]  # Set segmented regions to red with full opacity

# Overlay the segmented regions on the original image
overlay_image = np.maximum(input_image_rgba, segmented_overlay)

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_image_np)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_segmented, cmap='gray')
plt.title('Segmentation Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay_image)
plt.title('Overlay_image')
plt.axis('off')

plt.show()
