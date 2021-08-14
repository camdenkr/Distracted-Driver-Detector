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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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
    y_true = []
    y_pred = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(total)
            print(predicted==labels)
            print((predicted == labels).sum())
            correct += (predicted == labels).sum().item()
            print(type(predicted))
            print(labels)
            y_true += labels
            y_pred += predicted
    bool_list = [y_true[i]==y_pred[i] for i in range(len(y_true))]
    print(sum(bool_list)/len(y_true))
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    disp.plot()
    plt.show()

    classwiseTable = {}
    for c in range(10):
        classwiseTable["total"] = y_true.count(c)
        classwiseTable["correct"] = [y_true[i]==y_pred[i] for i in range(len(y_true)) if y_true[i]==c].count(True)
        classwiseTable["incorrect"] = [y_true[i]==y_pred[i] for i in range(len(y_true)) if y_true[i]==c].count(False)
        classwiseTable["accuracy"] = classwiseTable["correct"]/classwiseTable["total"]*100
        print("c"+str(c)+"\n", classwiseTable)

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


history_model = r'vgg16'
history_epochs = r'5'

history_path = fr'/Users/keshav/Desktop/scripts/latest/Distracted-Driver-Detector/models/{history_model}/saved/{history_model}_history_saved_{history_epochs}batches'
# history_path = fr'/Users/keshav/Desktop/scripts/latest/Distracted-Driver-Detector/alex_noT'

def plotLoss():
    history = pd.read_pickle(history_path)

    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
    # for c in ['train_acc', 'valid_acc']:
        plt.plot(history[c], label=c)

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.show()

def plotAccuracy():
    history = pd.read_pickle(history_path)

    plt.figure(figsize=(8, 6))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(history[c], label=c)

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.show()

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
    num_epochs = 16
    model = get_model(model_choice)
    path = (str(Path(__file__).parent / "../models/" / model_choice / "saved" / str(str(model_choice) + "_saved_" + str(num_epochs) + "batches")))
    
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    test_model(model=model, test_loader=test_loader)

    plotLoss()
    plotAccuracy()


if __name__ == "__main__":
    main()
