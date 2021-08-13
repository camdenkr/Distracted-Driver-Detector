import sys
from pathlib import Path
import pandas as pd
import torch
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from distractedDriverDataset import AUCTestDataset
from get_model import get_model
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent)+"/..")


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
        for data in tqdm(test_loader):
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
    # Define testing data and loaders
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=226),
            transforms.CenterCrop(size=224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    path = str(Path(__file__).parent / "../data/AUC/v2_cam1_cam2_ split_by_driver/Camera 1")
    test_set = AUCTestDataset(csv_file = path + "/auc_test.csv", root_dir = path + '/test', transform = transform)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)
    model_choice = "googlenet"
    num_epochs = 6
    model = get_model(model_choice)
    path = (str(Path(__file__).parent / "../models/" / model_choice / "saved" / str(str(model_choice) + "_saved_" + str(num_epochs) + "batches")))
    
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # torch.load(model.state_dict(), path)
    test_model(model=model, test_loader=test_loader)


if __name__ == "__main__":
    main()