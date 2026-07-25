import torch._dynamo
torch._dynamo.disable()
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train = np.load("./data/x_train.npy")
t_train = np.load("./data/y_train.npy")

x_test = np.load("./data/x_test.npy")

class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        self.x_train = x_train.reshape(-1, 784).astype('float32') / 255
        self.t_train = t_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype = torch.float), torch.tensor(self.t_train[idx], dtype = torch.long)

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = x_test.reshape(-1, 784).astype('float32') / 255

    def __len__(self):
        return self.x_test.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_test[idx], dtype = torch.float)


trainval_data = train_dataset(x_train, t_train)
test_data = test_dataset(x_test)

## MLPの実装##

batch_size = 32

val_size = 10000
train_size = len(trainval_data) - val_size
train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size = batch_size,
    shuffle = True
)

dataloader_valid = torch.utils.data.DataLoader(
    val_data,
    batch_size = batch_size,
    shuffle = False
)

dataloader_test = torch.utils.data.DataLoader(
    test_data,
    batch_size = batch_size,
    shuffle = False
)

##活性化関数の定義##

def relu(x):
    x = torch.where(x > 0, x, torch.zeros_like(x))
    return x

def softmax(x):
    x -= x.max(axis = 1, keepdim = True).values
    x_exp = torch.exp(x)
    return x_exp / x_exp.sum(axis = 1, keepdim = True)

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, function = lambda x: x):
        super().__init__()
        self.W = nn.Parameter(torch.tensor(np.random.uniform(low = -np.sqrt(6 / in_dim), high = np.sqrt(6 / in_dim), size = (in_dim, out_dim)).astype('float32')))
        self.b = nn.Parameter(torch.tensor(np.zeros([out_dim]).astype('float32')))

        self.function = function

    def forward(self, x):
        return self.function(x @ self.W + self.b)

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()

        self.linear1 = Dense(in_dim, hid_dim, function = relu)
        self.linear2 = Dense(hid_dim, out_dim, function = softmax)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)

        return x

##実装##

in_dim = 784
hid_dim = 200
out_dim = 10
lr = 0.001
n_epochs = 10

mlp = MLP(in_dim, hid_dim, out_dim)

optimizer = optim.SGD(mlp.parameters(), lr = lr)

mlp = mlp.to(device)   

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    train_num = 0
    train_true_num = 0
    valid_num = 0
    valid_true_num = 0

    mlp.train()

    for x, t in dataloader_train:

        t_hot = torch.eye(10)[t]
        x = x.to(device)
        t_hot = t_hot.to(device)

        y = mlp(x)

        ##交差エントロピー誤差##
        loss = -(t_hot * torch.log(y)).sum(dim = 1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.tolist())
        pred = y.argmax(1)

        acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))

        train_num += acc.shape[0]
        train_true_num += acc.sum().item()

    mlp.eval()

    for x, t in dataloader_valid:
        t_hot = torch.eye(10)[t]
        x = x.to(device)
        t_hot = t_hot.to(device)

        with torch.no_grad():
            y = mlp(x)
            loss = -(t_hot * torch.log(y)).sum(dim=1).mean()

        losses_valid.append(loss.tolist())
        pred = y.argmax(1)

        acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
        valid_num += acc.size()[0]
        valid_true_num += acc.sum().item()

    print('EPOCH {}, Train [Loss : {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:3f}, Accuracy: {:3f}]'.format(
        epoch,
        np.mean(losses_train),
        train_true_num / train_num,
        np.mean(losses_valid),
        valid_true_num / valid_num
    ))








