# Distracted-Driver-Detector

Credit to:

https://github.com/WillKoehrsen/pytorch_challenge/blob/master/Transfer%20Learning%20in%20PyTorch.ipynb
https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
https://github.com/WillKoehrsen/pytorch_challenge/blob/master/Transfer%20Learning%20in%20PyTorch.ipynb


for training code

and 


To run code, call src/main.py with an argv of the name of the model choice (vgg16, alexNet, or googlenet) o/w the default will be vgg16. 


Models histories (for plots of loss and accuracy) are saved to: models/[model_choice]/saved/[model_choice]_history_saved
Models state dicts are saved to models/[model_choice]/saved/[model_choice]_saved


/data contains all the datasets
    /AUC contains one dataset for distracted drivers, of which we are only using test and train in data/AUC/v2_cam1_cam2_ split_by_driver/Camera 1
    /StateFarm contains the Kaggle state farm dataset of which we are using /train since /test is unlabeled

/models contains the saved, trained models as well as the history of validationa and training accuracy over time for each model

/src contains all the source code for running the program
/distractedDriverDattaset defines two custom datasets, one for the Statefarm dataset and onee for the AUC datasets as their respective csvs are different

