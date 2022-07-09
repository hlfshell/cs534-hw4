import os
import torch

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_PATH = "./dataset"
EPOCHS = 50
LEARNING_RATE = 0.1

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

labels = [folder for folder in os.listdir(DATASET_PATH)]
print(f"{len(labels)} labels detected: {labels}")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(DATASET_PATH, transform=preprocess)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

for epoch in range(0, EPOCHS):
    print(f"***** EPOCH {epoch+1} *****")

    running_loss = 0.0

    for batch_index, data in enumerate(dataloader):
        output = model(data)

        loss = criterion(output, )