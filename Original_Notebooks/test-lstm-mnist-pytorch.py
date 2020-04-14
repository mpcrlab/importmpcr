
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from PIL import Image
import time
import copy


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset

flipped = True

MODEL =  'lstm/lstm-100e-0.001lr-16bs-300hd-adamopt-0.9590000000000001'
TEST_IM = '9L.png'

NUM_CLASSES = 10
INPUT_DIM = 27
N_LAYERS = 1
HIDDEN_DIM = 300
BS = 1


im = Image.open(TEST_IM)
im = np.asarray(im.resize((28,28)))[:,:,0]
# im = np.asarray(im.resize((28,28)))


def load_batch(x):
    X_lengths = [len(x)]
    longest = 29
    ad = np.zeros((longest-x.shape[0], 28))
    ad[:, 27] = 1
    im = np.append(x, ad, axis=0)
    ins = torch.from_numpy(np.expand_dims(im, 0)).to(torch.float)
    return ins, X_lengths

def normies(bin):
    mean = np.mean(np.abs(bin))
    std = np.std(bin)
    norm = bin-mean
    norm = norm/std
    return norm


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
        hidden1 = (weight.new(N_LAYERS, BS, HIDDEN_DIM).zero_().to(torch.int64),
                  weight.new(N_LAYERS, BS, HIDDEN_DIM).zero_().to(torch.int64))
        return hidden1



htsteps = []
ctsteps = []
for tstep in range(1, im.shape[0]+1):
    model = torch.load(MODEL, map_location=lambda storage, loc: storage)
    model.eval()
    h1 = model.init_hidden1(N_LAYERS, BS)
    ins, X_length = load_batch(im[:tstep, :])
    outs, hid = model(ins, X_length)
    h = np.sum(np.abs(hid[0].detach().numpy()))
    c = np.sum(np.abs(hid[1].detach().numpy()))
    htsteps.append(h)
    ctsteps.append(c)
correct = str(np.argmax(outs.detach().numpy()))+'-'
print('preds:', outs, '@ tstep: ', tstep, 'final pred:', correct)
cnorm = normies(np.abs(np.asarray(ctsteps)))
hnorm = normies(np.abs(np.asarray(htsteps)))
plt.bar(np.arange(cnorm.shape[0]), cnorm, width=1)
plt.title('MNIST Normalized Cell State Activation per Timestep ')
plt.savefig('lstm/abs_sum_figs/'+correct+TEST_IM+'MNIST_Normalized_Cell_State_Activation_per_Timestep ' + '.png')
plt.close()
plt.bar(np.arange(hnorm.shape[0]), hnorm, width=1)
plt.title('MNIST Normalized Hidden State Activation per Timestep ')
plt.savefig('lstm/abs_sum_figs/'+correct+TEST_IM+ 'MNIST_Normalized_H_State_Activation_per_Timestep ' + '.png')
plt.close()


f = open('lstm/abs_sum_figs/' + TEST_IM + '-all_hidden.txt', 'w')
f.write(str(htsteps))
f.close()




##
f = open('lstm/abs_sum_figs/9.png-all_hidden.txt')
sr = f.read()
sr = sr.split('[')[1]
sr = sr.split(']')[0]
sr = np.asarray(sr.split(','), dtype='float64')
nine = normies(sr)

f = open('lstm/abs_sum_figs/9L.png-all_hidden.txt')
sr = f.read()
sr = sr.split('[')[1]
sr = sr.split(']')[0]
sr = np.asarray(sr.split(','), dtype='float64')
ninel = normies(sr)

f = open('lstm/abs_sum_figs/9R.png-all_hidden.txt')
sr = f.read()
sr = sr.split('[')[1]
sr = sr.split(']')[0]
sr = np.asarray(sr.split(','), dtype='float64')
niner = normies(sr)

f = open('lstm/abs_sum_figs/7.png-all_hidden.txt')
sr = f.read()
sr = sr.split('[')[1]
sr = sr.split(']')[0]
sr = np.asarray(sr.split(','), dtype='float64')
sevn = normies(sr)

f = open('lstm/abs_sum_figs/7L.png-all_hidden.txt')
sr = f.read()
sr = sr.split('[')[1]
sr = sr.split(']')[0]
sr = np.asarray(sr.split(','), dtype='float64')
sevnl = normies(sr)

f = open('lstm/abs_sum_figs/7R.png-all_hidden.txt')
sr = f.read()
sr = sr.split('[')[1]
sr = sr.split(']')[0]
sr = np.asarray(sr.split(','), dtype='float64')
sevnr = normies(sr)

width = .35
fig, ax = plt.subplots()
rects1 = ax.bar(np.arange(28) - width/2, (nine-niner), width, label='nine-nine_right')
rects2 = ax.bar(np.arange(28) + width/2, (sevn-niner), width, label='seven-nine_right')
# rects3 = ax.bar(np.arange(28), sevn, width, label='seven')
ax.legend()
ax.set_ylabel('Activation')
ax.set_xlabel('Time Step (Column)')
ax.set_title('Differences in Hidden State Activations')
# plt.show()
plt.savefig('9-9R_7-9R.png')

print(np.sum(np.abs(nine)-np.abs(niner)), np.sum(np.abs(sevn)-np.abs(niner)))
print(np.sum(nine-niner), np.sum(sevn-niner))









#
