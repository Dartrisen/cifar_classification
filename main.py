import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import Network
from prepare_data import CustomDataset, transformations

batch_size = 100
number_of_labels = 3

# Create train and test datasets
train_set = CustomDataset(data_dir=os.getcwd() + r"/train", transform=transformations)
test_set = CustomDataset(data_dir=os.getcwd() + r"/test", transform=transformations)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

print("The number of images in a training set is: ", len(train_loader)*batch_size)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('label_0', 'label_1', 'label_2')


# Instantiate a neural network model
model = Network(number_of_labels)

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


# Function to save the model
def save_model():
    path = "./model.pth"
    torch.save(model.state_dict(), path)


# Function to test the model with the test dataset and print the accuracy for the test images
def test_accuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    accuracy = (100 * accuracy / total)
    return accuracy


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("The model will be running on", device, "device")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        accuracy = test_accuracy()
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        if accuracy > best_accuracy:
            save_model()
            best_accuracy = accuracy


# Function to show the images
def image_show(img):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def test_batch():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    image_show(torchvision.utils.make_grid(images))
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    train(10)
    print('Finished Training')

    accuracy = test_accuracy()
    print(accuracy)

    model = Network(number_of_labels)
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)

    test_batch()
