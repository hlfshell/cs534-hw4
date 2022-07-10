import os
import torch

import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_PATH = "./dataset_300"
FULL_DATASET_PATH = "./dataset_full"
EPOCHS = 25
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
validation_dataset = datasets.ImageFolder(FULL_DATASET_PATH, transform=preprocess)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

lowest_validation_loss = 1_000_000
losses = []
validation_losses = []

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
        losses.append(loss.item())

        # Status update every 10 batches
        if batch_index % 10 == 9 or batch_index == 0:    
            print(f'Epoch: {epoch + 1}, Batch: {batch_index+1}, Avg. Loss: {running_loss/(batch_index+1)}')

    # After every epoch, let's check to see how it stacks up against
    # the full dataset
    # Model into eval mode
    with torch.no_grad():
        model.eval()
        total_batches = len(validation_dataloader)
        total_loss = 0.0
        for _, data in enumerate(validation_dataloader):
            inputs, labels = data
            
            output = model(inputs)
            loss = criterion(output, labels)

            total_loss += loss.item()

    # Put us back into training mode
    model.train()
    validation_loss = total_loss / total_batches
    validation_losses.append(validation_loss)
    print(f"Validation test loss - {validation_loss}")
    if validation_loss < lowest_validation_loss:
        # Save the model!
        lowest_validation_loss = validation_loss
        filename = f'model-{epoch+1}.pt'
        model_path = os.path.join("./", filename)
        torch.save(model.state_dict(), model_path)
        print(f'New validation loss low -  trained model saved to {filename}')

print("***** Training complete! *****")
final_model_path = "model-final.pt"
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved as {final_model_path}")

plt.figure("losses")
plt.plot(losses, linestyle = 'dotted')
plt.title("Training Loss Over Time")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig("./loss_plot.jpg")

plt.figure("validation_losses")
plt.plot(validation_losses, linestyle = 'dotted')
plt.title('Validation Training Loss Over Time')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("./validation_loss_plot.jpg")