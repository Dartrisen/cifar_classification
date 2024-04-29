import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


# Master dict for outputs?
label_dict = {
    "healthy": 0,
    "DM": 1,
    "JAS": 2
}

# Loading and normalizing the data.
# Define transformations for training and test
transformations = transforms.Compose([
    # transforms.Resize((32, 32)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

batch_size = 3
number_of_labels = 3


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # List of image file paths
        self.images = []
        self.labels = []
        for filename in os.listdir(data_dir):
            # Get the full path of the file
            file_path = os.path.join(data_dir, filename)
            print("file_path: ", file_path)
            # Check if the path is a file (not a directory)
            if os.path.isdir(file_path):
                for image_name in os.listdir(file_path):
                    name_list = image_name.split('__')
                    image_label = name_list[1].split('.')[0]
                    print("name list: ", name_list)
                    print("image label: ", image_label)

                    image_path = os.path.join(data_dir, filename, image_name)
                    if os.path.isfile(image_path):
                        # Append the file path to the list
                        self.images.append(image_path)
                        self.labels.append(label_dict[image_label])
                        print("image, label: ", image_path, label_dict[image_label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Create train and test datasets
train_set = CustomDataset(data_dir=os.getcwd() + r"/train", transform=transformations)
test_set = CustomDataset(data_dir=os.getcwd() + r"/test", transform=transformations)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

print("The number of images in a training set is: ", len(train_loader)*batch_size)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('DM', 'healthy', 'JAS')
