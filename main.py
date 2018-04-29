import numpy as np
from utils import get_timit_dict, get_target, pretty_spectrogram, sliding_window, get_batch_data, TimitDataset
from model import CapsuleNet, CapsuleLoss
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import copy

# initialize parameters
labels = get_timit_dict('phonedict.txt')
nr_classes = len(labels)
batch_size = 64

rate = 16000 #16000fps - 0.0625ms per frame
stepsize = 64 #for spectogram reduction

frame_size = (int)((0.030 * rate) / stepsize) #30ms
frame_step = (int)((0.015 * rate) / stepsize) #15ms

print('Frame size: {}, frame step size: {}'.format(frame_size, frame_step))

# preprocess data
dataset = TimitDataset('./data', labels, stepsize, frame_step, frame_size)
dataloader = get_batch_data(dataset, batch_size)

device = torch.cuda.device(0)

model = CapsuleNet(num_clases=nr_classes)
model.cuda()

capsule_loss = CapsuleLoss()
optimizer = optim.Adam(model.parameters())

def train_model(model, optimizer, num_epochs=10):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs, reconstructions = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = capsule_loss(inputs, labels, outputs, reconstructions)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = train_model(model, optimizer, num_epochs=25)