import numpy as np
from utils import get_timit_dict, get_batch_data, TimitDataset
from model import CapsuleNet, CapsuleLoss
import torch
from torch import optim
from torch.autograd import Variable
import copy
from sklearn.metrics import accuracy_score

# initialize parameters
labels = get_timit_dict('phonedict.txt')
nr_classes = len(labels)
batch_size = 64
epochs = 1

rate = 16000 #16000fps - 0.0625ms per frame
stepsize = 64 #for spectogram reduction
freq_bins = 64

frame_size = (int)((0.025 * rate) / stepsize) #30ms
frame_step = (int)((0.010 * rate) / stepsize) #15ms

print('Frame size: {}, frame step size: {}'.format(frame_size, frame_step))

# preprocess data
dataset = TimitDataset('./data', labels, stepsize, freq_bins, frame_step, frame_size)
dataloader = get_batch_data(dataset, batch_size)

device = torch.cuda.device(0)

model = CapsuleNet(num_clases=nr_classes)
model.cuda()

capsule_loss = CapsuleLoss()
optimizer = optim.Adam(model.parameters())

def train_model(model, optimizer, num_epochs=10):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()

        for inputs, labels in dataloader:

            inputs = Variable(inputs).cuda()
            print(inputs.size())
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            outputs, reconstructions = model(inputs)
            _, preds = torch.max(outputs, 1)
            _, true = torch.max(labels, 1)

            loss = capsule_loss(inputs, labels, outputs, reconstructions)
            loss.backward()
            optimizer.step()

            print('Loss: {:.4f}'.format(
                (float)(loss.data)))

        print()

    return model

model = train_model(model, optimizer, num_epochs=epochs)