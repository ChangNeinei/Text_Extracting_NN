from __future__ import print_function, division

import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from skimage import io, transform
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import scipy.io

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
hidden_size = 64
train_data = scipy.io.loadmat('data/nist36_train.mat')
valid_data = scipy.io.loadmat('data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
train_x = torch.from_numpy(train_x).float()
train_y = train_y.argmax(axis = 1)
train_y = torch.from_numpy(train_y)

model = torch.nn.Sequential(
    torch.nn.Linear(1024, hidden_size),
    torch.nn.BatchNorm1d(hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, 36),
    torch.nn.BatchNorm1d(36),
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
max_iters = 50

def get_random_batches(x,y,batch_size):
    data_size = x.shape[0]
    batch_index = np.random.permutation(data_size)
    batches = []
    for i in range(data_size//int(batch_size)):
        index = batch_index[i * batch_size: i * batch_size + batch_size]
        mini = []
        mini.append(x[index])
        mini.append(y[index])
        batches.append(mini)
    return batches

plot_loss = []
plot_acc = []
plot_v_loss = []
plot_v_acc = []
batches = get_random_batches(train_x, train_y, 60)

for t in range(max_iters):
    total_loss = 0
    total_acc = 0
    valid_loss = 0
    valid_acc = 0
    for xb, yb in batches:
    # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(xb)

    # Compute and print loss
        loss = criterion(y_pred, yb)
        total_loss = total_loss + loss
        p_label = torch.argmax(y_pred, dim = 1)
        acc = (yb == p_label).sum()
        total_acc = total_acc + acc / len(yb)
    # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    total_acc = total_acc / len(batches)
    plot_loss.append(total_loss)
    plot_acc.append(acc)
    

x_axis = np.arange(0, max_iters, 1)
plt.plot(x_axis, plot_loss, label = 'training loss')
plt.legend(loc='upper right', shadow=True)
plt.show()

x_axis = np.arange(0, max_iters, 1)
plt.plot(x_axis, plot_acc, label = 'training acc')
plt.legend(loc='upper right', shadow=True)
plt.show()