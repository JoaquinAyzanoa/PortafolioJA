from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import numpy as np
import torchvision as tv


train_mean = [0.5, 0.5, 0.5] #for 3 channels
train_std = [0.5, 0.5, 0.5]

#train_mean = [0.59685254]#for 1 channels
#train_std = [0.16043035]

import warnings
warnings.filterwarnings('ignore')

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self.img_size = 512

        # Define transformations based on the mode
        if self.mode == 'train':
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((self.img_size, self.img_size)),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomRotation(30),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])
        else:
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize((self.img_size, self.img_size)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load image and label from the dataframe
        image_path = Path(self.data.iloc[index, 0]) 
        label = np.array([float(self.data.iloc[index, 1]),
                          float(self.data.iloc[index, 2]),
                          float(self.data.iloc[index, 3]),
                          float(self.data.iloc[index, 4]),
                          float(self.data.iloc[index, 5]),
                          float(self.data.iloc[index, 6]),
                          float(self.data.iloc[index, 7])])

        # Load the image and convert to RGB if it's grayscale
        #image = imread(image_path, as_gray = True)
        image = imread(image_path)
        #image = np.float32(image)
        #print(image)
        if len(image.shape) == 2:
            image = gray2rgb(image)
        #if len(image.shape) == 3:
        #    image = rgb2gray(image)

        # Apply transformations
        image = self.transform(image)
        

        label = torch.tensor(label, dtype=torch.float)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        image = image.to(device)
        #print(label)
        label = label.to(device)
        
        return image, label
    

