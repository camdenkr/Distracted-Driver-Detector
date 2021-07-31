import sys, os
# import pandas as pd
from pathlib import Path
import torch
# import torchvision
from torch.utils.data import DataLoader #manages dataset and creates mini batches
# from tqdm import tqdm
import torchvision.transforms as transforms
from distractedDriverDataset import DistractedDriverDataset
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent)+"/..")
from models.cnn_test import TestNet

# Hyperparameters:
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 1

# Load Data
DATA_PATH = str(Path(__file__).parent / "../data/StateFarm")
# DATA_PATH = '../data/StateFarm'
# Percent of training:validation set
# #load data:
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((48,48)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = DistractedDriverDataset(csv_file = DATA_PATH + '/driver_imgs_list.csv', root_dir = DATA_PATH + '/imgs/train', transform = transform)
#Split randomly:
train_set, test_set = torch.utils.data.random_split(dataset, [20000, 2424])


train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)



# Model
# we initialize our model as thus:
# Congrats! You just built a neural network with PyTorch :-)
net = TestNet()


# GPU-aware programming
"""
our PyTorch module loads automatically to CPU, and afterwards we can decide to
send it to GPU using .to() method. In fact tensor.to() method can send any
PyTorch tensor to any device, not just models.
"""
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(DEVICE)
net.to(DEVICE)  # this sends the model to the appropriate device

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

path = str(Path(__file__).parent / "testnet_saved")
torch.save(net.state_dict(), path)
