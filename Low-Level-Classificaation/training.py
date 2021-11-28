import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
from torch.nn import init
import torchaudio.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
from matplotlib import pyplot as plt


def open(path):
    sig, sr = torchaudio.load(path)
    return sig, sr

def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

def make_data():
    data = np.load('data/input_final.npy')
    labels = np.load('data/labels_final.npy')

    audio_input, audio_labels = [], []
    audio_data = []

    cnt = 0
    for idx, sample in tqdm(enumerate(data), total=data.shape[0]):
        sample_num = sample[-1]
        if os.path.exists(f'data/audio/sample_{sample_num}.mp3'):
            path = f'data/audio/sample_{sample_num}.mp3'
            aud = open(path)
            sig, sr = aud
            total = sig.shape[1]
            sig = sig[:, total // 2 : total // 2 + 441600]
            spec = spectro_gram(aud)
            if(spec.shape[2] != 2588):
                continue
            audio_data.append(spec.numpy())
            audio_input.append(sample)
            audio_labels.append(labels[idx])
    
    print(f'Saving audio_input')
    np.save('data/audio_input.npy', np.array(audio_input))
    print(f'Saving audio_labels')
    np.save('data/audio_labels.npy', np.array(audio_labels))
    print(f'Saving audio_data')
    np.save('data/audio_data.npy', np.array(audio_data))

    print(f'Count = {cnt}')


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

def training():
    model = AudioClassifier()
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

    print(device)

    model = model.to(device)

    np.random.seed(0)
    audio_data = np.load('data/audio_data.npy')
    audio_labels = np.load('data/audio_labels.npy')

    perm = np.random.permutation(audio_data.shape[0])
    num_train = int(0.8 * audio_data.shape[0])
    num_test = audio_data.shape[0] - num_train
    
    X_train, X_test = audio_data[perm[:num_train]], audio_data[perm[num_train:]]
    Y_train, Y_test = audio_labels[perm[:num_train]], audio_labels[perm[num_train:]]

    if not os.path.exists('split/X_train.npy'):
        np.save('split/X_train.npy', X_train)
        np.save('split/Y_train.npy', Y_train)
        np.save('split/X_test.npy', X_test)
        np.save('split/Y_test.npy', Y_test)

    ds = SoundDS(X_train, Y_train)


    print(num_train)
    print(num_test)
    print(num_train + num_test)
    train_ds = ds


    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    num_epochs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
    

    losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

    
        for i, (audio, label) in tqdm(enumerate(train_dl), total=len(train_dl)):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = audio.to(device), label.to(device)
            labels = labels.view(-1)
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        # if (epoch + 1) % 5 == 0 or epoch + 1 == 100:
        #     torch.save(model.state_dict(), f'models/model_{epoch}.pt')

    plt.plot(losses)
    plt.title('Loss plot')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'plots/loss.png')
    print('Finished Training')




if __name__ == '__main__':
    # make_data()
    training()