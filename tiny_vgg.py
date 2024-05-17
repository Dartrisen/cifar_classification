import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from prepare_data import CustomDataset

print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# set up hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 100
NUM_EPOCHS = 10
NUMBER_OF_LABELS = 3

transformations = transforms.Compose([
    transforms.Resize(size=IMAGE_SIZE),
    transforms.ToTensor()
])

# Create train and test datasets
train_set = CustomDataset(data_dir=os.getcwd() + r"/train", transform=transformations)
test_set = CustomDataset(data_dir=os.getcwd() + r"/test", transform=transformations)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("The number of images in a training set is: ", len(train_loader)*BATCH_SIZE)
print("The number of images in a test set is: ", len(test_loader)*BATCH_SIZE)

print("The number of batches per epoch is: ", len(train_loader))


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.
    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    if device:
        print(f"\nTrain time using PyTorch device {device}: {total_time:.3f} seconds\n")
    else:
        print(f"\nTrain time: {total_time:.3f} seconds\n")
    return round(total_time, 3)


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_shape: int, output_shape: int) -> None:
        super(TinyVGG, self).__init__()
        self.feature_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_shape, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape, out_channels=hidden_shape, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.feature_block_2 = nn.Sequential(
            nn.Conv2d(hidden_shape, hidden_shape, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_shape, hidden_shape, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_shape * 53 * 53, out_features=output_shape)  # image shape 32, 32
        )

    def forward(self, x: torch.Tensor):
        x = self.feature_block_1(x)
        x = self.feature_block_2(x)
        x = self.classifier(x)
        return x


model = TinyVGG(input_shape=3, hidden_shape=10, output_shape=3)
print(model(torch.randn(1, 3, 224, 224)))


def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
):
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device
):
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            test_pred_logits = model(x)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
