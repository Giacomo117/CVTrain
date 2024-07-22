import pandas as pd
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.models import resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt

# Definition of useful config variables
base_directory = r"C:\Users\Tomma\PycharmProjects\FacialDetection\new_data"
output_directory = r"C:\Users\Tomma\PycharmProjects\FacialDetection\output"
batch_size = 32
epochs = 100


class FaceKeypointDataset(Dataset):
    def __init__(self, samples, path):
        self.data = samples
        self.path = path
        self.resize = 224

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Reading the image and converting it to Color Space
        image = cv2.imread(f"{self.path}/{self.data.iloc[index, 0]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Retrieve images dimensions
        orig_h, orig_w, channel = image.shape

        # Resize the image into `resize` defined above.
        image = cv2.resize(image, (self.resize, self.resize))

        # Image normalization
        image = image / 255.0

        # Transpose for getting the channel size to index 0.
        image = np.transpose(image, (2, 0, 1))

        # Get the keypoints.
        keypoints = self.data.iloc[index][1:]
        keypoints = np.array(keypoints, dtype='float32')

        # Reshape the keypoints.
        keypoints = keypoints.reshape(-1, 2)

        # Rescale keypoints according to image resize.
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }


# Loading the csv files
training_samples = pd.read_csv(os.path.join(base_directory, 'training.csv'))
valid_samples = pd.read_csv(os.path.join(base_directory, 'test.csv'))

# Initialize the dataset `FaceKeypointDataset()`
train_data = FaceKeypointDataset(training_samples, os.path.join(base_directory, 'training'))
valid_data = FaceKeypointDataset(valid_samples, os.path.join(base_directory, 'test'))

# Prepare data loaders
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)
valid_loader = DataLoader(valid_data,
                          batch_size=batch_size,
                          shuffle=False)

# Loading the ResNet50 model with pre-trained weights
model = resnet50(weights='DEFAULT')

# Unfreeze all parameters for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Change the final layer to match keypoints dimensions
model.fc = nn.Linear(in_features=2048, out_features=136)

# ?????
model = model.to('cpu')

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Set loss function.
criterion = nn.SmoothL1Loss()


# Define training function.
def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0

    for i, data in tqdm(enumerate(dataloader), total=len(train_loader)):
        counter += 1
        image, keypoints = data['image'].to('cpu'), data['keypoints'].to('cpu')
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


# Define validation function.
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            counter += 1
            image, keypoints = data['image'].to('cpu'), data['keypoints'].to('cpu')
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()

    valid_loss = valid_running_loss / counter
    return valid_loss


# Start of training loop
train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    # Call of training methods
    train_epoch_loss = fit(model, train_loader, train_data)
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)

    # Update loss arrays for plotting
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')

# Creation of loss plots images
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{output_directory}/loss.png")
plt.show()

# Saving the computed model
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{output_directory}/model.pth")