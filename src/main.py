import sys, os
# import pandas as pd
import random
from pathlib import Path
import torch
# import torchvision
from torch.utils.data import DataLoader #manages dataset and creates mini batches
import torchvision.transforms as transforms
from distractedDriverDataset import DistractedDriverDataset
from distractedDriverDataset import AUCTestDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from timeit import default_timer as timer
from torchvision import models
import pandas as pd
sys.path.append(str(Path(__file__).parent)+"/..")
import matplotlib.pyplot as plt
from torch import optim, cuda

# Hyperparameters:
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 17
train_on_gpu = False

''' 
Loads and returns models
-------
model_choice: name of a model to be loaded
'''
def get_model(model_choice):
    if(model_choice == "vgg16"):
        print("vgg16")
        model = models.vgg16(pretrained=True)
        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features
        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10), nn.LogSoftmax(dim=1))
        
    if(model_choice == "alexNet"):
        print("alexNet")
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10), nn.LogSoftmax(dim=1))
    if(model_choice == "googlenet"):
        print("googlenet")
        model = models.googlenet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10), nn.LogSoftmax(dim=1))
    return model


'''
Trains a model
------
model: model object to be trained
train_loader: training set loader
'''
def train_model(model, model_choice, train_loader, valid_loader, criterion, optimizer):
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    overall_start = timer()

    # Main loop
    for epoch in range(num_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        train_on_gpu = cuda.is_available()
        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

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

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % 1 == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    path = (str(Path(__file__).parent / "../models/" / model_choice / "saved" / str(str(model_choice) + "_saved_" + str(num_epochs) + "batches")))
                    torch.save(model.state_dict(), path)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history



def main():
    # Load Data
    DATA_PATH = str(Path(__file__).parent / "../data/StateFarm")

    # Transforms not including data augmentations
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=226),
            transforms.CenterCrop(size=224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    # Transforms for training data, including data augmentations
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=226),
            transforms.CenterCrop(size=224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        



    # Find unique drivers in statefarm dataset
    DATA_PATH = str(Path(__file__).parent / "../data/StateFarm")
    df = pd.read_csv(DATA_PATH + '/driver_imgs_list.csv')
    by_drivers = df.groupby("subject")
    unique_drivers = list(by_drivers.groups.keys())
    # Set validation set percentage with regards to training set
    val_pct = 0.2
    random.shuffle(unique_drivers)
    # These are the drivers we will be entirely moving to the validation set
    to_val_drivers = unique_drivers[:int(len(unique_drivers) * val_pct)]
    
    #First split validation and training images for StateFarm dataset based on unique drivers
    df = pd.read_csv(DATA_PATH + '/driver_imgs_list.csv', index_col=False)
    matched_df = df.loc[df['subject'].isin(to_val_drivers)]
    unmatched_df = df.loc[~df['subject'].isin(to_val_drivers)]
    matched_df.to_csv(DATA_PATH + "/SF_val.csv", index=False, header=["subject", "classname", "img"])
    matched_df.to_csv(DATA_PATH + "/SF_val_pers.csv", index=False, header=["subject", "classname", "img"])
    unmatched_df.to_csv(DATA_PATH + "/SF_train.csv", index=False, header=["subject", "classname", "img"])
    matched_df.to_csv(DATA_PATH + "/SF_train_pers.csv", index=False, header=["subject", "classname", "img"])
    
    #SF data
    SF_train_set = DistractedDriverDataset(csv_file = DATA_PATH + '/SF_train.csv', root_dir = DATA_PATH + '/imgs/train', transform = train_transform)
    val_set = DistractedDriverDataset(csv_file = DATA_PATH + '/SF_val.csv', root_dir = DATA_PATH + '/imgs/train', transform = test_transform)
    
    #AUC data
    path = str(Path(__file__).parent / "../data/AUC/v2_cam1_cam2_ split_by_driver/Camera 1")
    test_set = AUCTestDataset(csv_file = path + "/auc_test.csv", root_dir = path + '/test', transform = test_transform)
    AUC_train_set = AUCTestDataset(csv_file = path + "/auc_train.csv", root_dir = path + '/train', transform = train_transform)

    # Combine SF and AUC training sets
    train_set = torch.utils.data.ConcatDataset([AUC_train_set, SF_train_set])

    #Set up loaders
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    valid_loader= DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    #Get a model as a commmand line argument, default to vgg16 if no argument is given
    if (len(sys.argv) != 2):
        model_choice = "vgg16"
    else:
        model_choice = sys.argv[1]
    
    model = get_model(model_choice)
    model.epochs = 1

    # GPU-aware programming
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(DEVICE)
    model.to(DEVICE)

    # Train model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    model, history = train_model(model, model_choice, train_loader, valid_loader, criterion, optimizer)
    
    #Save the history of validation and training accuracies and losses
    path = (str(Path(__file__).parent / "../models/" / model_choice / "saved" / str(str(model_choice) + "_history_saved_" + str(num_epochs) + "batches")))
    history.to_pickle(path)


if __name__ == "__main__":
    main()