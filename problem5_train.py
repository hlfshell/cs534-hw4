import os
import torch

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_PATH = "./dataset_300"
EPOCHS = 50
LEARNING_RATE = 0.01

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

# Freeze the weights in the model
for param in model.parameters():
    param.requires_grad = False

labels = [folder for folder in os.listdir(DATASET_PATH)]
print(f"{len(labels)} labels detected: {labels}")

# Prepare the new FCN layer
in_features = model.classifier[-1].in_features
out_features = len(labels)
# Remove the last layer of our model
features = list(model.classifier.children())[:-1]
#... and add in a new one that outputs to our number of categories
features.extend([torch.nn.Linear(in_features, out_features)])
# Finally assign it back
model.classifier = torch.nn.Sequential(*features)

# Preprocess layer to get images to match input of Alexnet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(DATASET_PATH, transform=preprocess)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(0, EPOCHS):
    print(f"***** EPOCH {epoch+1} *****")

    running_loss = 0.0

    for batch_index, data in enumerate(dataloader):
        inputs, labels = data

        output = model(inputs)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Status update every 10 batches
        if batch_index % 10 == 9 or batch_index == 0:    
            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_index+1, running_loss/(batch_index+1)))