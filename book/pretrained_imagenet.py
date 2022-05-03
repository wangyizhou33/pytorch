from torchvision import models

print(">> Pytorch has the following built-in networks")
print(dir(models))

alexnet = models.AlexNet()

print(">> Alexnet has the following architecture: ")
print(alexnet)

resnet = models.resnet101(pretrained=True)
print(">> resnet has the following architecture: ")
print(resnet)

# preprocessing function
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# run inference
from PIL import Image

print(">> Run inference of the input bobby.jpg with resnet.")
img = Image.open("./bobby.jpg")
img_t = preprocess(img)

import torch

batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)

print(">> Output tensor is ")
print(out)

with open("./imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(">> resnet predicted: ")
    print(labels[index[0]], percentage[index[0]].item())

    _, indices = torch.sort(out, descending=True)
    print(">> The network also predicted:")
    for idx in indices[0][1:5]:
        print(labels[idx], percentage[idx].item())