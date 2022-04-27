from pickle import TRUE
import torch
import torchvision
import torchvision.transforms as transforms

# download the data
# options are: CIFAR10, MNIST, FashionMNIST
train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)

# print some important information of the dataset
print(">> Number of samples in the dataset")
print(len(train_set))

print(">> Labels of the set")
print(train_set.targets)

print(">> Bincount of the set")
# print(train_set.targets.bincount())

# inspect the sample
sample = next(iter(train_set))

print(
    ">> Each sample has length 2. The first is the image tensor. The second is the label/target tensor."
)
print(len(sample))
print(type(sample))

# sequence unpacking
# image = sample[0]
# label = sample[1]
image, label = sample
print(">> The image has the following shape:")
print(image.shape)
print(">> Label has no shape because it is an int. The value is")
print(label)

# visualize the sample
plt.imshow(image.squeeze(), cmap="gray")
plt.savefig(fname="sample" + ".png", format="png")

# inspect the batch
train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
batch = next(iter(train_loader))
print(
    ">> Each batch has length 2 again. The first is the image tensor (aggregated). The second is the label/target tensor."
)
print(len(batch))
print(type(batch))

# sequence unpacking
# image = sample[0]
# label = sample[1]
images, labels = batch
print(">> The images tensor has the following shape:")
print(images.shape)  # [10, 1, 28, 28]
print(labels.shape)  # [10]

# visualize the batch
grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))
plt.savefig(fname="batch" + ".png", format="png")
