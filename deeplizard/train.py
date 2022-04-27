from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        #  (1) input layer
        t = t

        # (2) conv layer 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) conv layer 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) fully-connected layer 1
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) fully-connected layer 2
        t = self.fc2(t)
        t = F.relu(t)

        # (6) fully-connected output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)

        return t


network = Network()

print(">> The network has the following architecture: ")
print(network)

print(">> The network has the init weights: ")
print(network.conv1.weight)

for name, param in network.named_parameters():
    print(name, "\t\t", param.shape)


# prepare data to train
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

torch.set_grad_enabled(False)

# predict a single sample
sample = next(iter(train_set))
image, label = sample  # image has shape torch.Size([1, 28, 28])

# forward pass
pred = network(
    image.unsqueeze(0)
)  # unsqueeze gives output [1, 1, 28, 28], i.e. batch size 1

print(">> Predict a sample now")
print(">> Output tensor with softmax looks like")
print(F.softmax(pred, dim=1))
print(
    ">> Before the training, the prediction is ",
    pred.argmax(dim=1),
    ", while the target is ",
    label,
)

#  predict a batch
train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
batch = next(iter(train_loader))
images, labels = batch  # image has shape torch.Size([1, 28, 28])

# forward pass
preds = network(images)  # unsqueeze gives output [1, 1, 28, 28], i.e. batch size 1
print(">> Predict a batch now")
print(
    ">> Before the training, the prediction is ",
    preds.argmax(dim=1),
    ", while the target is ",
    labels,
)
print(">> Prediction accuracy", preds.argmax(dim=1).eq(labels))
