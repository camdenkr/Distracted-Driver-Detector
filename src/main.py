import sys, os
# import pandas as pd
from pathlib import Path
import torch
# import torchvision
from torch.utils.data import DataLoader #manages dataset and creates mini batches
import torchvision.transforms as transforms
from distractedDriverDataset import DistractedDriverDataset
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timeit import default_timer as timer
from torchvision import models

sys.path.append(str(Path(__file__).parent)+"/..")
from models.cnnTest.cnn_test import TestNet

# Hyperparameters:
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 1

''' 
Loads and returns models
-------
model_choice: name of a model to be loaded
'''
def get_model(model_choice):
    if(model_choice == "vgg16"):
        model = models.vgg16(pretrained=True)

    # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 10), nn.LogSoftmax(dim=1))
    return model

'''
Trains a model
------
model: model object to be trained
train_loader: training set loader
'''
def train_model(model, train_loader, criterion, optimizer):

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    model.to(DEVICE)

    


    for epoch in range(num_epochs):

            # keep track of training and validation loss each epoch
            train_loss = 0.0
            valid_loss = 0.0

            train_acc = 0
            valid_acc = 0

            # Set to training
            model.train()
            start = timer()

            # Training loop
            for ii, (data, target) in enumerate(train_loader):
            

                # Clear gradients
                optimizer.zero_grad()
                # Predicted outputs are log probabilities
                output = model(data)

                # Loss and backpropagation of gradients
                loss = criterion(output, target)
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Track train loss by multiplying average loss by number of examples in batch
                train_loss += loss.item() * data.size(0)

                # Calculate accuracy by finding max log probability
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                # Need to convert correct tensor from int to float to average
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples in batch
                train_acc += accuracy.item() * data.size(0)

                # Track training progress
                print(
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')

'''
Tests a trained model
'''
def test_model(model, test_loader):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))


    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))



def main():
    # Load Data
    DATA_PATH = str(Path(__file__).parent / "../data/StateFarm")

    #https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
    #https://github.com/WillKoehrsen/pytorch_challenge/blob/master/Transfer%20Learning%20in%20PyTorch.ipynb

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    dataset = DistractedDriverDataset(csv_file = DATA_PATH + '/driver_imgs_list.csv', root_dir = DATA_PATH + '/imgs/train', transform = transform)

    #Split randomly:
    #TODO: Split the dataset by driver
    # We can split the datset by generating 2 dataset csvs and getting them with two objects

    train_set, test_set = torch.utils.data.random_split(dataset, [20000, 2424])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)



    #Get a model
    model_choice = "vgg16"
    model = get_model(model_choice)
    
    # Train model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    train_model(model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)
    
    #Save model
    path = str(Path(__file__).parent / "../models/vgg16/vgg_test_saved")
    #! Do not uncomment
    # torch.save(model.state_dict(), path)

    # Load model
    model.load_state_dict(torch.load(path))
    test_model(model, test_loader=test_loader)

    

if __name__ == "__main__":
    main()