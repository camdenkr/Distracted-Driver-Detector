# Distracted-Driver-Detector


All code was run using python 3.8.6 and PyTorch 1.8.1

Code can be run by calling src/main.py with a command line argument of either "vgg16" "alexNet" or "googlenet" to run training on the respective pretrained models. If no argument is given, training will default to vgg16. Trained models and pickled model hisotries are saved to the following paths:

Models histories (for plots of loss and accuracy) are saved to: models/[model_choice]/saved/[model_choice]_history_saved
Models state dicts are saved to models/[model_choice]/saved/[model_choice]_saved

Test.py contains the code in order to test the accuracy of a given model by loading the models from the path listed above.



/data contains all the datasets*
    /AUC contains one dataset for distracted drivers, of which we are only using test and train in data/AUC/v2_cam1_cam2_ split_by_driver/Camera 1
    /StateFarm contains the Kaggle state farm dataset of which we are using /train since /test is unlabeled

/models* contains the saved, trained models as well as the history of validationa and training accuracy over time for each model

/src contains all the source code for running the program

/distractedDriveraDattaset defines two custom datasets, one for the Statefarm dataset and onee for the AUC datasets as their respective csvs are different

/outputs contains various, unorganized, stdout of the program running over various attempts.


*Trained models, histories, and datasets were not uploaded for conservation of space. 


Credit to:
Code:
https://github.com/WillKoehrsen/
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
https://stackabuse.com/image-classification-with-transfer-learning-and-pytorch/

Datasets:
https://www.kaggle.com/carloalbertobarbanovgg16-transfer-learning-pytorch
https://abouelnaga.io/projects/auc-distracted-driver-dataset/
