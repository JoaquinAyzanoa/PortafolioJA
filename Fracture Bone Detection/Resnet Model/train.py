import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from prepare_data import *

device = 'cuda:0' if t.cuda.is_available() else 'cpu'
print(device)

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules

train_data, _ = prepare_data('.\data', 'train')
#print(train_data)
val_data, _ = prepare_data('.\data', 'valid')
#data = pd.read_csv("data.csv", delimiter=';')  # Update with your actual file path
#train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(train_data, mode='train')
val_dataset = ChallengeDataset(val_data, mode='val')
train_dataloader = t.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True)
val_dataloader = t.utils.data.DataLoader(val_dataset, batch_size=48, shuffle=False)

# create an instance of our ResNet model
resnet_model = model.ResNet()
resnet_model.to(device)
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.CrossEntropyLoss()  # Assuming it's a multi class classification task
optimizer = t.optim.Adam(resnet_model.parameters(), lr=0.00001)
trainer = Trainer(model=resnet_model, crit=criterion, optim=optimizer,
                  train_dl=train_dataloader, val_test_dl=val_dataloader,
                  cuda=True, early_stopping_patience=25)

# go, go, go... call fit on trainer
#trainer.restore_checkpoint(79)
res = trainer.fit(epochs=200)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')