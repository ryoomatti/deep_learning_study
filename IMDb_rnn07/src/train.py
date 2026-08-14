import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import string
import re
from typing import List, Union

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

##データ準備##

x_train = np.load('./data/x_train.npy', allow_pickle = True)
t_train = np.load('./data/t_train.npy', allow_pickle = True)

x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size = 0.2, random_state = seed)

def text_transform(text: List[int], max_length = 256):
    text = text[:max_length - 1] + [2]

    return text, len(text)

def collate_batch(batch):
    label_list, text_list, len_seq_list = [], [], []

    for sample in batch:
        if isinstance(sample, tuple):
            label, text = sample

            label_list.append(label)
        else:
            text = sample.copy()

        text, len_seq = text_transform(text)
        text_list.append(torch.tensor(text))
        len_seq_list.append(len_seq)

    return torch.tensor(label_list), pad_sequence(text_list, padding_value = 3).T, torch.tensor(len_seq_list)

word_num = np.concatenate(np.concatenate((x_train, x_valid))).max() + 1
print(f"Vocabulary size: {word_num}")

##実装##

batch_size = 128

train_dataloader = DataLoader(
    [(t, x) for t, x in zip(t_train, x_train)],
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate_batch,
)

valid_dataloader = DataLoader(
    [(t, x) for t, x in zip(t_valid, x_valid)],
    batch_size = batch_size,
    shuffle = False,
    collate_fn = collate_batch
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##train関数の定義##
def train(net, optimizer, n_epochs,):
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []

        net.train()
        n_train = 0
        acc_train = 0
        for label, line, len_seq in train_dataloader:
            net.zero_grad()  # Reset gradients

            t = label.to(device)  # Move tensors to GPU
            x = line.to(device)  # (batch, time)
            len_seq.to(device)

            h = net(x, torch.max(len_seq), len_seq)
            y = torch.sigmoid(h).squeeze()

            loss = -torch.mean(t * torch_log(y) + (1 - t) * torch_log(1 - y))  # WRITE ME

            loss.backward()  # Backpropagate the loss

            optimizer.step()  # Update parameters

            losses_train.append(loss.tolist())

            n_train += t.size()[0]

        # Valid
        t_valid = []
        y_pred = []
        net.eval()
        for label, line, len_seq in valid_dataloader:

            t = label.to(device)  # Move tensors to GPU
            x = line.to(device)
            len_seq = len_seq.to(device)

            h = net(x, torch.max(len_seq), len_seq)
            y = torch.sigmoid(h).squeeze()

            loss = -torch.mean(t * torch_log(y) + (1 - t) * torch_log(1 - y))  # WRITE ME

            pred = y.round().squeeze()  # Predict the positive label for values >= 0.5

            t_valid.extend(t.tolist())
            y_pred.extend(pred.tolist())

            losses_valid.append(loss.tolist())

        print('EPOCH: {}, Train Loss: {:.3f}, Valid Loss: {:.3f}, Validation F1: {:.3f}'.format(
            epoch,
            np.mean(losses_train),
            np.mean(losses_valid),
            f1_score(t_valid, y_pred, average='macro')
        ))

def torch_log(x):
    return torch.log(torch.clamp(x, min = 1e-10))

##Embedding層##
class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding_matrix = nn.Parameter(
            torch.randn(
                (vocab_size, emb_dim),
                dtype = torch.float
            )
        )
    def forward(self, x):
        return F.embedding(x, self.embedding_matrix)
##LSTMの実装##
class LSTM(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        glorot = 6 / (in_dim + hid_dim * 2)

        self.W_i = nn.Parameter(torch.tensor(np.random.uniform(
            low = -np.sqrt(glorot),
            high = np.sqrt(glorot),
            size = (in_dim + hid_dim, hid_dim)
        ).astype('float32')))

        self.b_i = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

        self.W_f = nn.Parameter(torch.tensor(np.random.uniform(
            low = -np.sqrt(glorot),
            high = np.sqrt(glorot),
            size = (in_dim + hid_dim, hid_dim)
        ).astype('float32')))

        self.b_f = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

        self.W_o = nn.Parameter(torch.tensor(np.random.uniform(
            low=-np.sqrt(glorot),
            high=np.sqrt(glorot),
            size=(in_dim + hid_dim, hid_dim)
        ).astype('float32')))
        self.b_o = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

        self.W_c = nn.Parameter(torch.tensor(np.random.uniform(
            low=-np.sqrt(glorot),
            high=np.sqrt(glorot),
            size=(in_dim + hid_dim, hid_dim)
        ).astype('float32')))
        self.b_c = nn.Parameter(torch.tensor(np.zeros([hid_dim]).astype('float32')))

    def function(self, state_c, state_h, x):
        i = torch.sigmoid(torch.matmul(torch.cat([state_h, x], dim = 1), self.W_i) + self.b_i)
        f = torch.sigmoid(torch.matmul(torch.cat([state_h, x], dim = 1), self.W_f) + self.b_f)
        o = torch.sigmoid(torch.matmul(torch.cat([state_h, x], dim = 1), self.W_o) + self.b_o)
        c = f * state_c + i * torch.tanh(torch.matmul(torch.cat([state_h, x], dim = 1), self.W_c) + self.b_c)
        h = o * torch.tanh(c)  
        return c, h

    def forward(self, x, len_seq_max = 0, init_state_c = None, init_state_h = None):
        x = x.transpose(0, 1)
        state_c = init_state_c
        state_h = init_state_h

        if init_state_c is None:
            state_c = torch.zeros((x[0].size()[0], self.hid_dim)).to(x.device)

        if init_state_h is None:
                    state_h = torch.zeros((x[0].size()[0], self.hid_dim)).to(x.device)
        

        size = list(state_h.unsqueeze(0).size())
        size[0] = 0

        output = torch.empty(size, dtype = torch.float).to(x.device)

        if len_seq_max == 0:
            len_seq_max = x.size(0)
        for i in range(len_seq_max):
            state_c, state_h = self.function(state_c, state_h, x[i])
            output = torch.cat([output, state_h.unsqueeze(0)])
        return output


class SequenceTaggingNet(nn.Module):
    def __init__(self, word_num, emb_dim, hid_dim):
        super().__init__()
        self.emb = Embedding(word_num, emb_dim)
        self.lstm = LSTM(emb_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, 1)

    def forward(self, x, len_seq_max = 0, len_seq = None, init_state_c = None, init_state_h = None):
        h = self.emb(x)
        h = self.lstm(h, len_seq_max, init_state_c, init_state_h)
        if len_seq is not None:
            h = h[len_seq - 1, list(range(len(x))), :]
        else:
            h = h[-1]

        y = self.linear(h)

        return y

emb_dim = 100
hid_dim = 50
n_epochs = 5

net = SequenceTaggingNet(word_num, emb_dim, hid_dim)
net.to(device)
optimizer = optim.Adam(net.parameters())

train(net, optimizer, n_epochs)
    


