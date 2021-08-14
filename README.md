# Distracted-Driver-Detector

Distracted driving is one the leading causes of car accidents in thw world. This includes activities such as applying makeup, eating, or texting while driving. However, deep learning techniques have made it possible to predict whether a driver is adequately paying attention while on the road. We wxplored the use of transfer learning with 3 different CNN architectures that are able to predict whether a given image shows a driver who is distracted while driving with varying degrees of accuracy from 20% to 70%. The architectures used are VGG16, GoogLeNet, and AlexNet. Among these, GoogLeNet was the most accurate when trained with no frozen layers.



## Running Code
All code was run using python 3.8.6 and PyTorch 1.8.1  

The repo may be downloaded as is. Data must be added for the training code to run properly. The Kaggle dataset must be downloaded from https://www.kaggle.com/c/state-farm-distracted-driver-detection and unzipped and placed into the path: /data/StateFarm. The AUC must be downloaded by emailing researchers through https://abouelnaga.io/projects/auc-distracted-driver-dataset/ and a direct link cannot be publicly provided. It must then be placed in /data/AUC.

In order to test a trained model, test.py may be run, but the corresponding file path to a trained must be changed within the file, and should be stored as described further down.



Code can be run by calling src/main.py with a command line argument of either "vgg16" "alexNet" or "googlenet" for example:
```python3 src/main.py vgg16```
to run training on the respective pretrained models. If no argument is given, training will default to vgg16. Trained models and pickled model hisotries are saved to the following paths:

Models histories (for plots of loss and accuracy) are saved to: models/[model_choice]/saved/[model_choice]_history_saved
Trained Model state dicts are saved to models/[model_choice]/saved/[model_choice]_saved

Test.py contains the code in order to test the accuracy of a given model by loading the models from the path listed above. This file also contains code to plot the trining and vvalidation accuracies and losses.


## Structure
/data contains all the datasets*
    /AUC contains one dataset for distracted drivers, of which we are only using test and train in data/AUC/v2_cam1_cam2_ split_by_driver/Camera 1
    /StateFarm contains the Kaggle state farm dataset of which we are using /train since /test is unlabeled

/models* contains the saved, trained models as well as the history of validationa and training accuracy over time for each model

/src contains all the source code for running the program
    distractedDriveraDattaset.py defines two custom datasets, one for the Statefarm dataset and onee for the AUC datasets as their respective csvs are different
    main.py is the main code used to train various models
    test.py is used to plot the history of validationa and training accuracies and losses over time as well as compute the testing accuracy for a given trained model
    get_model.py returns a specified, pretrained model used for transfer learning: vgg16, googlenet, or alexnet
/outputs contains various, unorganized, stdout of the program running over various attempts.


*Trained models, histories, and datasets were not uploaded for conservation of space. 


### Presentation Slides:
[CS523 Group5.pdf](https://github.com/camdenkr/Distracted-Driver-Detector/files/6985972/CS523.Group5.pdf)


### Paper:
[CS523_report.pdf](https://github.com/camdenkr/Distracted-Driver-Detector/files/6985875/CS523_report.pdf)

Credit to:
Code:
https://github.com/WillKoehrsen/
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
https://stackabuse.com/image-classification-with-transfer-learning-and-pytorch/
https://www.kaggle.com/carloalbertobarbanovgg16-transfer-learning-pytorch
Datasets:
https://www.kaggle.com/c/state-farm-distracted-driver-detection
https://abouelnaga.io/projects/auc-distracted-driver-dataset/
