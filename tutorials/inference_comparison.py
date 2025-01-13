import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import random
import onnx
import onnxruntime as ort
import numpy as np

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load("model-113.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

random_integer = random.randint(1, 100)


model.eval()
with torch.no_grad():
    for i in range(10):
        sample_idx = random_integer + i
        x, y = test_data[sample_idx][0], test_data[sample_idx][1]
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


# convert pytorch to onnx
input_tensor = test_data[0][0]

# Export the model
torch.onnx.export(
    model,                        # PyTorch model
    input_tensor,                 # Example input tensor
    "model-113.onnx",          # Output file name
    export_params=True,           # Store the trained parameter weights
    opset_version=11,             # ONNX version (use at least 11 for compatibility)
    do_constant_folding=True,     # Optimize constant folding
    input_names=["input"],        # Specify input names
    output_names=["output"],      # Specify output names
    dynamic_axes={                # Specify axes that are dynamic
        "input": {0: "batch_size"}, 
        "output": {0: "batch_size"}
    }
)
print("Model has been converted to ONNX format.")

# Load the ONNX model
session = ort.InferenceSession("model-113.onnx")

for i in range(10):
    sample_idx = random_integer + i
    x, y = test_data[sample_idx][0], test_data[sample_idx][1]
    # Perform inference
    onnx_output = session.run(None, {"input": x.numpy()})   # convert tensor to numpy array
    pred = torch.from_numpy(onnx_output[0])                 # convert numpy array to tensor
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

