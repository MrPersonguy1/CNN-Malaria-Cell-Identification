# -*- coding: utf-8 -*-
"""
# Imports
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import numpy as np

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchsummary import summary

"""
#Data Loading
"""

!pip install --upgrade kagglehub

import kagglehub

path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")

print("Path to dataset files:", path)

print(os.listdir(os.path.join(path, 'cell_images')))

uninfected_cells = os.path.join(path, 'cell_images', 'Uninfected')
parasitized_cells = os.path.join(path, 'cell_images', 'Parasitized')

print(os.listdir(uninfected_cells))

cell_to_num = {
    0: 'Uninfected',
    1: 'Parasitized'
}

print(f'Uninfected Cell Images: {len(os.listdir(uninfected_cells))}')
print(f'Parasitized Cell Images: {len(os.listdir(parasitized_cells))}')

def load_image_label(image_folder, label, image_size=(64, 64)):
    images = []
    labels = []

    # Creating a list of files in directory
    file_list = os.listdir(image_folder)

    for i in tqdm(range(len(file_list)), desc="Loading Data"):
        # Index the file from the list
        filename = file_list[i]
        # print(f'Filename: {filename}')

        # Construct the full path
        file_path = os.path.join(image_folder, filename)
        # print(f'File Path: {file_path}')
        # input()

        # Turning the path into an imgage
        image = cv2.imread(file_path)


        # Skip if image couldn't be loaded
        if image is None:
            print(f"Skipping {file_path} - could not be loaded")
            continue

        # Resize the image to 64, 64
        image = cv2.resize(image, image_size)

        # Convert from BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to 0-1
        image = image / 255.0

        # Append to lists
        images.append(image)
        labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

uninfected_images, uninfected_label = load_image_label(uninfected_cells, 0)
parasitized_images, parasitized_label = load_image_label(parasitized_cells, 1)

print(f'Uninfected Images: {uninfected_images.shape}')
print(f'Uninfected Labels: {uninfected_label.shape}')
print(f'Parasitized Images: {parasitized_images.shape}')
print(f'Parasitized Labels: {parasitized_label.shape}')

images = np.concatenate((uninfected_images, parasitized_images), axis=0)

labels = np.concatenate((uninfected_label, parasitized_label), axis=0)

print(images.shape)

"""#Data Visualizing"""

image_num = 9234 #@param {type:"raw"}
plt.imshow(images[image_num])
plt.title(f'Label: {cell_to_num[...]}')
plt.show()

"""
# Create the Model
"""

class CNN(nn.Module):
  def __init__(self):
    # CODE HERE
    super().__init__()

    self.conv_layers = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.fully_connected = nn.Sequential(
        nn.Linear(in_features=64*8*8, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=2),
    )

    self.flatten = nn.Flatten()

  def forward(self, x):
    x = self.conv_layers(x)
    x = self.flatten(x)
    x = self.fully_connected(x)
    return x

model = CNN()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)
summary(model, (3, 64, 64))

"""
#Create the dataloaders
"""

print(images.shape)
print(labels.shape)

all_images = torch.tensor(images, dtype=torch.float32)

all_images = all_images.permute(0, 3, 1, 2)

all_labels = torch.tensor(labels, dtype=torch.long)

print(all_images.shape)
print(all_labels.shape)

train_images = all_images[:20000]
train_labels = all_labels[:20000]
test_images = all_images[20000:]
test_labels = all_labels[20000:]

train_set = TensorDataset(train_images, train_labels)
test_set = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

train_iter = iter(train_loader)
images, labels = next(train_iter)
print(images.shape)
print(labels.shape)
print(f"# of batches: {len(train_loader)}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

"""
# Train Loop
"""

def train_loop(train_dataloader, model, loss_fn, optimizer, epochs):
  train_loss = []

  for epoch in range(epochs):
    train_loss_epoch = 0

    for image, label in tqdm(train_dataloader, desc="Training Model"):
      image, label = image.to(device), label.to(device)
      optimizer.zero_grad()
      pred = model(image)
      loss = loss_fn(pred, label)
      loss.backward()
      train_loss_epoch += loss.item()
      optimizer.step()

    avg_loss = train_loss_epoch / len(train_dataloader)
    train_loss.append(avg_loss)

    print(f'Epoch: {epoch+1} | Loss: {avg_loss:.4f}')

  return train_loss

num_epochs = 5
losses = train_loop(train_loader, model, loss_fn, optimizer, epochs=num_epochs)

print(losses)

epoch_list = list(range(1, num_epochs+1))
plt.plot(epoch_list, losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

"""
#Creating the Testing Function
"""

def accuracy(correct, total):
  return correct/total * 100

def test_loop(test_dataloader, model):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for image, label in tqdm(test_dataloader, desc="Testing Model"):
      image, label = image.to(device), label.to(device)
      pred = model(image)
      correct += (pred.argmax(1) == label).type(torch.float).sum().item()
      total += len(label)

    print(f'Accuracy: {accuracy(correct, total)}')

test_loop(test_loader, model)

"""
# Visualize a prediction with the trained model
"""

rand_idx = 7557 #@param {type:"raw"}
testing_image, testing_label = test_set[rand_idx]

with torch.no_grad():
  pred = model(testing_image.unsqueeze(0))
  print(pred.shape)

plt.imshow(testing_image.cpu().permute(1, 2, 0))
plt.title(f'Prediction: {cell_to_num[pred.argmax(1).item()]} | Actual: {cell_to_num[testing_label.item()]}')
plt.show()
