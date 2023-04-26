import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Model
class SimpleNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(SimpleNet, self).__init__()
        self.N = 32 * 32
        self.linear1 = nn.Linear(in_features=32 * 32, out_features=self.N)
        self.linear2 = nn.Linear(in_features=self.N, out_features=self.N)
        self.linear3 = nn.Linear(in_features=self.N, out_features=self.N)
        self.linear4 = nn.Linear(in_features=self.N, out_features=num_classes)

    def forward(self, x):
        # Assume input is already flattened
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return  x

# Dataset

class MNISTloader:
    def __init__(
        self,
        batch_size: int = 100,
        data_dir: str = "./data/",
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        train_val_split: float = 0.1,
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.train_val_split = train_val_split

        self.setup()

    def setup(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.train_dataset = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        val_split = int(len(self.train_dataset) * self.train_val_split)
        train_split = len(self.train_dataset) - val_split

        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [train_split, val_split]
        )
        self.test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )

        print(
            "Image Shape:    {}".format(self.train_dataset[0][0].numpy().shape),
            end="\n\n",
        )
        print("Training Set:   {} samples".format(len(self.train_dataset)))
        print("Validation Set: {} samples".format(len(self.val_dataset)))
        print("Test Set:       {} samples".format(len(self.test_dataset)))

    def load(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        return train_loader, val_loader, test_loader

# Train + evaluate

def evaluate(device, model, criterion, val_loader):

    val_loss_running, val_acc_running = 0, 0

    model.eval().cuda() if (device.type == "cuda") else model.eval().cpu()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, dim=1)
            val_loss_running += loss.item() * inputs.shape[0]
            val_acc_running += torch.sum(predictions == labels.data)

        val_loss = val_loss_running / len(val_loader.sampler)
        val_acc = val_acc_running / len(val_loader.sampler)

    return val_loss, val_acc


def train(num_epochs, model, optimizer, criterion, train_loader, device):

    model.train().cuda() if (device.type == "cuda") else model.train().cpu()

    for epoch in range(num_epochs):

        train_loss_running, train_acc_running = 0, 0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predictions = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_running += loss.item() * inputs.shape[0]
            train_acc_running += torch.sum(predictions == labels.data)

        train_loss = train_loss_running / len(train_loader.sampler)
        train_acc = train_acc_running / len(train_loader.sampler)

        info = "Epoch: {:3}/{} \t train_loss: {:.3f} \t train_acc: {:.3f}"
        print(info.format(epoch + 1, num_epochs, train_loss, train_acc))
