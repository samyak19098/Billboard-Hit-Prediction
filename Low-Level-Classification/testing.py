import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn


class SoundDS(Dataset):
    def __init__(self, audio_data, audio_labels):
        self.audio_data = audio_data
        self.audio_labels = audio_labels

    def __len__(self):
        return self.audio_labels.shape[0]
    
    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        label = self.audio_labels[idx]
        # if label[0] == 0:
        #     label = np.array([0, 1])
        # else:
        #     label = np.array([1, 0])

        return audio, label
    
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=2)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

X_test, Y_test = torch.from_numpy(np.load('split/X_test.npy')), torch.from_numpy(np.load('split/Y_test.npy'))
Y_test = Y_test.flatten()
Y_test = Y_test.numpy()
print(X_test[0].shape)

def calc_curves(Y_test, prediction_proba, model_name, plot=False):
    ra_score = roc_auc_score(Y_test, prediction_proba)
    fpr, tpr, thresholds_roc = roc_curve(Y_test, prediction_proba)
    if plot:
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.savefig(f'../models/{model_name}/plots/roc_curve_{model_name}.png', facecolor='w', bbox_inches='tight')
        # plt.show()
        plt.close()
    print(f'RA score = {ra_score}')
    print(f'best threshold : {thresholds_roc[np.argmax(tpr - fpr)]}')
for model_num in range(29, 30, 1):

    model = AudioClassifier()
    model.load_state_dict(torch.load(f'models/model_{model_num}.pt'))

    output = model(X_test)
    _, prediction = torch.max(output, 1)

    prediction = prediction.numpy()

    tp = ((prediction == Y_test) & (prediction == 1)).sum()
    tn = ((prediction == Y_test) & (prediction == 0)).sum()
    fp = ((prediction != Y_test) & (prediction == 1)).sum()
    fn = ((prediction != Y_test) & (prediction == 0)).sum()

    stats = {}
    stats['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    stats['precision'] = (tp) / (tp + fp)
    stats['recall'] = (tp) / (tp + fn)
    stats['f1'] = 2 * (stats['recall'] * stats['precision']) / (stats['recall'] + stats['precision'])
    matrix = confusion_matrix(Y_test, prediction)

    seaborn.heatmap(matrix / np.sum(matrix), fmt='.2%', cmap='Blues', annot=True)
    plt.xlabel('Predicted value')
    plt.ylabel('Actual value')
    plt.title('Confusion matrix')
    plt.savefig(f'confusion_matrix.png', facecolor='w', bbox_inches='tight')
    # plt.show()
    plt.close()

    correct = (prediction == Y_test).sum()
    total = Y_test.shape[0]

    print(f'Model {model_num}: stats = {stats}')



# prediction = model()