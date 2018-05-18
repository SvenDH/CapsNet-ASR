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
epochs = 10

rate = 16000 #16000fps - 0.0625ms per frame
stepsize = 64 #for spectogram reduction
freq_bins = 32

frame_size = (int)((0.060 * rate) / stepsize) #30ms
frame_step = (int)((0.030 * rate) / stepsize) #15ms

print('Frame size: {}, frame step size: {}'.format(frame_size, frame_step))

# preprocess data
training_set = TimitDataset('./data', labels, stepsize, freq_bins, frame_step, frame_size)
trainloader = get_batch_data(training_set, batch_size)

test_set = TimitDataset('./data', labels, stepsize, freq_bins, frame_step, frame_size, traintest='TEST')
testloader = get_batch_data(training_set, batch_size)

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
        running_loss = 0.0

        for idx, (inputs, labels) in enumerate(trainloader, 0):

            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            outputs, reconstructions = model(inputs)
            _, preds = torch.max(outputs, 1)
            _, true = torch.max(labels, 1)

            loss = capsule_loss(inputs, labels, outputs, reconstructions)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if idx % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, idx + 1, running_loss / 2000))
                running_loss = 0.0

            #print('Loss: {:.4f}'.format(

            #    (float)(loss.data)))

        print()

    return model

def test_model(model):

    correct = 0.0
    total = 0.0

    for inputs, labels in testloader:

        outputs, reconstructions = model(inputs)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

model = train_model(model, optimizer, num_epochs=epochs)
test_model(model)