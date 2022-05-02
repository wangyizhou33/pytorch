from pickletools import optimize
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


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

torch.set_grad_enabled(True)

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

# loss function
loss = F.cross_entropy(preds, labels)
print(">> The initial loss is ", loss.item())

#  backward propagation
print(">> Initially there is no gradient. For example the conv1 layer has")
print(network.conv1.weight.grad)
loss.backward()
print(">> After the back propagation, the same layer has")
print(network.conv1.weight.grad.shape)

import torch.optim as optim

# updating the weights
optimizer = optim.Adam(network.parameters(), lr=0.01)
print(">> Before the step, the loss is ", loss.item())
print("The network gets ", get_num_correct(preds, labels), " image correct")
optimizer.step()
preds = network(images)
loss = F.cross_entropy(preds, labels)
print(">> After the step, the loss is ", loss.item())
print("The network gets ", get_num_correct(preds, labels), " image correct")

# training loop
print(">> Now we train the network continuously.")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(5):
    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()  # calculate gradients
        optimizer.step()  # update the weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print("epoch: ", epoch, " total_correct: ", total_correct, " loss ", total_loss)

print(">> Accuracy after ", 5, " epochs: ", total_correct / len(train_set))

# visualizing the result via confusion matrix

# get the predictions in a single tensor
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


print(">> Predict the all set one more time and stack the results together.")
prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
train_preds = get_all_preds(network, prediction_loader)

stacked = torch.stack((train_set.targets, train_preds.argmax(dim=1)), dim=1)
print(">> The stack has shape ", stacked.shape)

# compute the confusion matrix
# similar to a histogram
cmt = torch.zeros(10, 10, dtype=torch.int64)

for p in stacked:
    j, k = p.tolist()
    cmt[j, k] = cmt[j, k] + 1

print(">> This is the confusion matrix")
print(cmt)
