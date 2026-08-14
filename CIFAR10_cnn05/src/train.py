import random

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

x_train = np.load('./data/x_train.npy')
t_train = np.load('./data/t_train.npy')

class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        data = x_train.astype('float32')
        self.x_train = []
        for i in range(data.shape[0]):
            self.x_train.append(Image.fromarray(np.uint8(data[i])))
        self.t_train = t_train
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.transform(self.x_train[idx]), torch.tensor(self.t_train[idx], dtype = torch.long)


trainval_data = train_dataset(x_train, t_train)

def fix_seed(seed = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

fix_seed(seed = 42)

##画像の前処理##

class gcn():
    def __init__(self):
        pass

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean) / (std + 10**(-6))

##ZCAWhiteningはペースト##

class ZCAWhitening():
    def __init__(self, epsilon=1e-4, device="cuda"):  # Since the calculation is heavy, use GPU
        self.epsilon = epsilon
        self.device = device

    def fit(self, images):  # Compute transformation matrix and mean from data
        x = images[0][0].reshape(1, -1)
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)
        for i in range(len(images)):  # Take the average for each data
            x = images[i][0].reshape(1, -1).to(self.device)
            self.mean += x / len(images)
            con_matrix += torch.mm(x.t(), x) / len(images)
            if i % 10000 == 0:
                print("{0}/{1}".format(i, len(images)))
        con_matrix -= torch.mm(self.mean.t(), self.mean)
        self.E, self.V = torch.linalg.eigh(con_matrix)  # Eigenvalue decomposition
        self.E = torch.max(self.E, torch.zeros_like(self.E)) # Preventing negative values ​​due to errors
        self.ZCA_matrix = torch.mm(torch.mm(self.V, torch.diag((self.E.squeeze()+self.epsilon)**(-0.5))), self.V.t())
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(tuple(size))
        x = x.to("cpu")
        x = x.float()
        return x

zca = ZCAWhitening(device = "cpu")
zca.fit(trainval_data)

val_size = 3000
train_data, val_data = torch.utils.data.random_split(trainval_data, [len(trainval_data) - val_size, val_size])

transforms_train = transforms.Compose([
    transforms.ToTensor(),
    gcn(),
    zca,
])

batch_size = 100

dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size = batch_size,
    shuffle = True
)

dataloader_valid = torch .utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle = False
)

import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

rng = np.random.RandomState(1234)
random_state = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##畳み込み層##

class Conv(nn.Module):
    def __init__(self, filter_shape, function = lambda x: x, stride = (1, 1), padding = 0):
        super().__init__()

        fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
        fan_out = filter_shape[0] * filter_shape[2] * filter_shape[3]

        self.W = nn.Parameter(torch.tensor(rng.normal(
            0,
            np.sqrt(2/fan_in),
            size = filter_shape
        ).astype('float32')))

        self.b = nn.Parameter(torch.tensor(np.zeros((filter_shape[0]), dtype = 'float32')))

        self.function = function
        self.stride = stride
        self.padding = padding   

    def forward(self, x):
        u = F.conv2d(x, self.W, bias = self.b, stride = self.stride, padding = self.padding)
        return self.function(u)

##平均プーリング層##

class Pooling(nn.Module):
    def __init__(self, ksize = (2, 2), stride = (2, 2), padding = 0):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size = self.ksize, stride = self.stride, padding = self.padding)

##平滑化層##

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)

##全結合層##

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, function = lambda x: x):
        super().__init__()

        self.W = nn.Parameter(torch.tensor(rng.normal(
            0,
            np.sqrt(2/in_dim),
            size = (in_dim, out_dim)
        ).astype('float32')))

        self.b = nn.Parameter(torch.tensor(np.zeros([out_dim]).astype('float32')))
        self.function = function

    def forward(self, x):
        return self.function(x @ self.W + self.b)

##活性化層##

class Activation(nn.Module):
    def __init__(self, function=lambda x: x):
        super().__init__()
        self.function = function

    def __call__(self, x):
        return self.function(x)  

##ネットワークの定義##

conv_net = nn.Sequential(
    Conv((32, 3, 3, 3)),
    Activation(F.relu),
    Pooling((2, 2)),
    Conv((64, 32, 3, 3)),
    Activation(F.relu),
    Pooling((2, 2)),
    Conv((128, 64, 3, 3)),
    Activation((F.relu)),
    Pooling((2, 2)),
    Flatten(),
    Dense(2 * 2 * 128, 256, F.relu),
    Dense(256, 10)

)

n_epochs = 10
lr = 0.01

conv_net.to(device)
optimizer = optim.Adam(conv_net.parameters(), lr = lr)

##学習##

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []

    conv_net.train()
    n_train = 0
    acc_train = 0
    for x, t in dataloader_train:
        n_train += t.size()[0]

        conv_net.zero_grad()

        x = x.to(device)

        t_hot = torch.eye(10)[t]
        t_hot = t_hot.to(device)

        y = conv_net.forward(x)

        loss = -(t_hot * torch.log_softmax(y, dim = -1)).sum(axis = 1).mean()
        loss.backward()
        optimizer.step()
        pred = y.argmax(1)
        acc_train += (pred == t).float().sum().item()
        losses_train.append(loss.tolist())

    conv_net.eval()
    n_val = 0
    acc_val = 0
    for x, t in dataloader_valid:
        n_val += t.size()[0]

        x = x.to(device)

        t_hot = torch.eye(10)[t]
        t_hot = t_hot.to(device)

        y = conv_net.forward(x)

        loss = -(t_hot * torch.log_softmax(y, dim = -1)).sum(axis = 1).mean()

        pred = y.argmax(1)

        acc_val += (pred == t).float().sum().item()
        losses_valid.append(loss.tolist())

    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
        epoch,
        np.mean(losses_train),
        acc_train/n_train,
        np.mean(losses_valid),
        acc_val/n_val
    ))




