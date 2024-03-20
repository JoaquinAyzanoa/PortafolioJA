import torch as t
from data import ChallengeDataset
from trainer2 import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model3
import pandas as pd
from sklearn.model_selection import train_test_split
from prepare_data import *

device = 'cuda:0' if t.cuda.is_available() else 'cpu'
print(device)

# Load data and split
train_data, _ = prepare_data('.\data', 'train')
val_data, _ = prepare_data('.\data', 'valid')

# Data loading
train_dataset = ChallengeDataset(train_data, mode='train')
val_dataset = ChallengeDataset(val_data, mode='val')
train_dataloader = t.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True)
val_dataloader = t.utils.data.DataLoader(val_dataset, batch_size=48, shuffle=False)

# Model initialization
resnet_model = model3.ResNetWithSegmentation(num_classes=7)  # Adjusted for 7 classes
resnet_model.to(device)

# Loss criterion
criterion = t.nn.CrossEntropyLoss() # Adjust loss function for segmentation task

# Optimizer
optimizer = t.optim.Adam(resnet_model.parameters(), lr=0.00001)

# Trainer initialization
trainer = Trainer(model=resnet_model, crit=criterion, optim=optimizer,
                  train_dl=train_dataloader, val_test_dl=val_dataloader,
                  cuda=True, early_stopping_patience=25)

# Training
res = trainer.fit(epochs=200)

# Results visualization
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
