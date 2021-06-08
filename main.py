import torch
from torchvision import datasets
from torchvision import utils
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = utils.data.DataLoader(
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
    # plt.imshow()
    # plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next() # type: ignore

# show images
# imshow(utils.make_grid(images))
utils.save_image(images / 2 + 0.5, "test.png")
# print labels
print(" ".join("%5s" % classes[labels[j]] for j in range(batch_size)))
