# -*- coding: utf-8 -*-
"""LSTM-pytorch

"""

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from PIL import Image
import progressbar
import glob
from urllib.request import Request, urlopen
from skimage.util import view_as_windows as vaw
import time
import copy
import visdom

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torchvision

vis = visdom.Visdom()



NUM_CLASSES = 10
N_LAYERS = 1
INPUT_DIM = 93
HIDDEN_DIM = 300

LR = .005
BS = 8
NUM_EPOCHS = 100
OPTIM = 'adam'

RESULTS = 'lstm'
PRESAVE_NAME = RESULTS + ('/lstm-'+str(NUM_EPOCHS)+'e-'+str(LR)+'lr-'+str(BS)+'bs-'+str(HIDDEN_DIM)+'hd-'+'ul-'+str(OPTIM)+'opt-')




"""load the data"""

train = torchvision.datasets.MNIST('toy-data' + '/', train=True, transform=torchvision.transforms.ToTensor(), download=True)

val = torchvision.datasets.MNIST('toy-data' + '/', train=False, transform=torchvision.transforms.ToTensor())

x = train.train_data
y = train.train_labels

xval = val.test_data
yval = val.test_labels



def load_batch(x, y):
    ins = []
    batch_idx = np.random.choice(x.shape[0], BS)
    batch_bin = x[batch_idx, ...]
    X_lengths = [len(sentence) for sentence in batch_bin]
    longest = 29
    for im in batch_bin:
        ad = np.zeros((longest-im.shape[0], 28))
        ad[:, 27] = 1
        im = np.append(im, ad, axis=0)
        ins.append(im)
    labels = y[batch_idx].to(torch.long)
    ins = torch.from_numpy(np.asarray(ins)).to(torch.float)
    # dataset = TensorDataset(ins, labels)
    return ins, labels, X_lengths

data = {'train': [x,y], 'val': [xval,yval]}


vis.image(x[0], win='ins')
vis.text(str(y[0]), win='labs')




''' model '''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Classifier_LSTM(nn.Module):
    def __init__(self, N_LAYERS, HIDDEN_DIM, BS):
        super(Classifier_LSTM, self).__init__()
        self.N_LAYERS = N_LAYERS
        self.HIDDEN_DIM = HIDDEN_DIM
        self.BS = BS
        # self.embed = nn.Embedding(27,27, padding_idx=26)
        self.lstm1 =  nn.LSTM(28, HIDDEN_DIM, num_layers=N_LAYERS, bias=True, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
    def forward(self, inputs, X_lengths):
        # e = self.embed(inputs)
        # e = e[:, :, :, 0]
        X = torch.nn.utils.rnn.pack_padded_sequence(inputs, X_lengths, batch_first=True, enforce_sorted=False)
        X, hidden1 = self.lstm1(X)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = X[:,-1,:]
        out = self.fc(X)
        return out, hidden1
    def init_hidden1(self, N_LAYERS, BS):
        weight = next(model.parameters()).data
        hidden1 = (weight.new(N_LAYERS, BS, HIDDEN_DIM).zero_().to(torch.int64).to(device),
                  weight.new(N_LAYERS, BS, HIDDEN_DIM).zero_().to(torch.int64).to(device))
        return hidden1

model = Classifier_LSTM(N_LAYERS, HIDDEN_DIM, BS)

model.to(device)


criterion = nn.CrossEntropyLoss()

if OPTIM == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
elif OPTIM == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)


def run():
    since = time.time()
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    best_acc = 0
    for epoch in progressbar.progressbar(range(NUM_EPOCHS)):
        h1 = model.init_hidden1(N_LAYERS, BS)
        for phase in ['train', 'val']:
            running_loss = 0
            running_corrects = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            x,y = data[phase]
            for i in range(x.shape[0]//BS):
                inputs, labels, X_lengths = load_batch(x,y)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outs, h = model(inputs, X_lengths)
                    _, preds = outs.max(1)
                    loss = criterion(outs, labels)
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(x)
            epoch_acc = running_corrects.double() / len(x)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_acc.append(epoch_acc.cpu().numpy())
                vis.line(train_acc, win='train_acc', opts=dict(title= '-train_acc'))
                train_loss.append(epoch_loss)
                vis.line(train_loss, win='train_loss', opts=dict(title= '-train_loss'))
            if phase == 'val':
                val_acc.append(epoch_acc.cpu().numpy())
                vis.line(val_acc, win='val_acc', opts=dict(title= '-val_acc'))
                val_loss.append(epoch_loss)
                vis.line(val_loss, win='val_loss', opts=dict(title= '-val_loss'))
    model.load_state_dict(best_model_wts)
    SAVE_NAME = PRESAVE_NAME + str(best_acc.detach().cpu().numpy())
    torch.save(model, SAVE_NAME)
    time_elapsed = time.time() - since
    val_loss_plt = plt.figure()
    plt.plot(val_loss)
    val_loss_plt.savefig(SAVE_NAME + '_val-loss.png')
    val_acc_plt = plt.figure()
    plt.plot(val_acc)
    val_acc_plt.savefig(SAVE_NAME + '_val-acc.png')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, val_acc, val_loss, best_acc, time_elapsed

model, val_acc, val_loss, best_acc, time_elapsed = run()








"""vestigual code"""
