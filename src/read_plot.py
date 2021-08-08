import pandas as pd

history = pd.read_pickle(r'/projectnb/cs542sp/kronhaus/Distracted-Driver-Detector/models/googlenet/saved/googlenet_history_saved')
import matplotlib.pyplot as plt


plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Negative Log Likelihood')
plt.title('Training and Validation Losses')