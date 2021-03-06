import os
import random
import pandas as pd
import torch
from skimage import io, transform
from sklearn import preprocessing
# from tqdm import tqdm
from torch.utils.data import Dataset
# import torchvision.transforms as transforms



class DistractedDriverDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        #Convert Labels into integers
        labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
        self.le = preprocessing.LabelEncoder()
        targets = self.le.fit_transform(labels)
        self.targets = torch.as_tensor(targets)

        

    # Returns the length of the dataset.
    def __len__(self):
        return len(self.df) # Should be ~22000 for SF
    
    # Returns a specific target image.    
    def __getitem__(self, index):
        # Pytorch sends in index, row index, col 2 (name of the image).
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 1], self.df.iloc[index, 2])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.le.transform([self.df.iloc[index, 1]])))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)



class AUCTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        #Convert Labels into integers
        labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
        self.le = preprocessing.LabelEncoder()
        targets = self.le.fit_transform(labels)
        self.targets = torch.as_tensor(targets)

        

    # Returns the length of the dataset.
    def __len__(self):
        return len(self.df) # Should be ~22000 for SF
    
    # Returns a specific target image.    
    def __getitem__(self, index):
        # Pytorch sends in index, row index, col 2 (name of the image).
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0], self.df.iloc[index, 1])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.le.transform([self.df.iloc[index, 0]])))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
