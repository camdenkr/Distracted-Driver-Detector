import pandas as pd
import random
import torch
import torchvision
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import Dataset

DATA_PATH = "./data/StateFarm"
df = pd.read_csv(DATA_PATH + "/driver_imgs_list.csv")
by_drivers = df.groupby("subject")
unique_drivers = by_drivers.groups.keys()
# Set validation set percentage with regards to training set
val_pct = 0.2
random.shuffle(unique_drivers)
# These are the drivers we will be entirely moving to the validation set
to_val_drivers = unique_drivers[:int(len(unique_drivers) * val_pct)]


class CustomDataset(Dataset):
    def __init__(self, csv_path, images_folder=, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = {"cat":0, "dog":1}

    def __len__(self):
        return len(self.df.img)
    
    def __getitem__(self, index):
        filename = self.df[index, "FILENAME"]
        label = self.class2index[self.df[index, "LABEL"]]
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label



SF_train_dataset = CustomDataset(DATA_PATH + "/driver_imgs_list.csv" , DATA_PATH + "/imgs/train"   )
# test_dataset = CustomDataset(DATA_PATH + "/driver_imgs_list.csv" , DATA_PATH + "/imgs/test"   )