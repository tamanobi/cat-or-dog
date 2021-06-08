import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.utils as utils
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    im = Image.fromarray(np.transpose(npimg, (1, 2, 0)))
    im.save("sample.png")


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()  # type: ignore

utils.save_image(images / 2 + 0.5, "test.png")
# print labels
print(" ".join("%5s" % classes[labels[j]] for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
