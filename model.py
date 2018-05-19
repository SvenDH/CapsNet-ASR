import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        #print(self.num_route_nodes)
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
            #print(self.route_weights.size())
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    @staticmethod
    def squash(tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = torch.matmul(x[None, :, :, None, :], self.route_weights[:, None, :, :, :])
            logits = Variable(torch.zeros(*priors.size()))
            if torch.cuda.is_available():
                logits = logits.cuda()
            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
            #print(outputs.size())

        return outputs

class CapsuleNet(nn.Module):
    def __init__(self, num_clases=10):
        super(CapsuleNet, self).__init__()
        self.num_clases = num_clases
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3,5), stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=3, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=self.num_clases, num_route_nodes=32 * 30, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * self.num_clases, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 240),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            if torch.cuda.is_available():
                y = Variable(torch.eye(self.num_clases)).cuda().index_select(dim=0, index=max_length_indices)
            else:
                y = Variable(torch.eye(self.num_clases)).index_select(dim=0, index=max_length_indices)
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True).double() ** 2
        right = F.relu(classes - 0.1, inplace=True).double() ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        reconstruction_loss = self.reconstruction_loss(reconstructions, images.view(images.size(0), -1)).double()

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, 5))
        #self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        #self.fc1 = nn.Linear(16 * 3 * 3, 128)
        #self.fc2 = nn.Linear(128, num_classes)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3,5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2).squeeze()
        #print(x.size())
        #x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        #x = x.view(-1, self.num_flat_features(x))
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        #return F.softmax(x, dim=-1)
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 4 * 4)
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.double()

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
