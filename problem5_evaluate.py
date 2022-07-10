import os
import torch

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_PATH = "./dataset_full"

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
labels = [folder for folder in os.listdir(DATASET_PATH)]

# Prepare the new FCN layer
in_features = model.classifier[-1].in_features
out_features = len(labels)
# Remove the last layer of our model
features = list(model.classifier.children())[:-1]
#... and add in a new one that outputs to our number of categories
features.extend([torch.nn.Linear(in_features, out_features)])
# Finally assign it back
model.classifier = torch.nn.Sequential(*features)

# Load the model state
model.load_state_dict(torch.load("./models/model_300_final.pt"))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


dataset = datasets.ImageFolder(DATASET_PATH, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model.eval()

total_images = 0
total_correct = 0

for _, data in enumerate(dataloader):
    inputs, labels = data

    output = model(inputs)
    out, inds = torch.max(output,dim=1)
    results = torch.sum(inds == labels).item()
    
    total_images += len(labels)
    total_correct += results

accuracy = total_correct / total_images

print(f"Final accuracy test of model is {accuracy:.2f}")
    