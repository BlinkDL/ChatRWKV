import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import math

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
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        seed_everything(42)
        
        self.N = 32 * 32
        self.linear1 = nn.Linear(in_features=self.N, out_features=self.N)
        self.linear2 = nn.Linear(in_features=self.N, out_features=self.N)
        self.linear3 = nn.Linear(in_features=self.N, out_features=self.N)
        self.linear4 = nn.Linear(in_features=self.N, out_features=num_classes)

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)

        residual = x

        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.relu(x) + residual
        x = self.linear3(x)
        x = F.relu(x) + residual
        x = self.linear4(x)        
        return  x
    
    def forward_pyquant(self, x):
        
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)

        x = self.quant(x)

        residual = x

        x = F.relu(self.linear1(x))        
        x = self.linear2(x)
        x = self.skip_add.add(F.relu(x), residual)
        x = self.linear3(x)
        x = self.skip_add.add(F.relu(x), residual)
        x = self.linear4(x)

        x = self.dequant(x)

        return x

class SimpleNet_V2(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet_V2, self).__init__()
        seed_everything(42)
        self.N = 32 * 32
        
        self.linear0_w = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.N, self.N), a=math.sqrt(5)))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.linear0_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        self.linear0_b = nn.Parameter(torch.nn.init.uniform_(torch.empty(self.N), -bound, bound))

        self.linear1_w = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.N, self.N), a=math.sqrt(5)))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.linear1_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        self.linear1_b = nn.Parameter(torch.nn.init.uniform_(torch.empty(self.N), -bound, bound))
        
        self.linear2_w = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.N, self.N), a=math.sqrt(5)))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.linear2_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        self.linear2_b = nn.Parameter(torch.nn.init.uniform_(torch.empty(self.N), -bound, bound))
        
        self.linear3_w = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(num_classes, self.N), a=math.sqrt(5)))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.linear3_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        self.linear3_b = nn.Parameter(torch.nn.init.uniform_(torch.empty(num_classes), -bound, bound))

        self.w = {}
        self.nb_layers = 0
        for i in range(0, 4):
            self.w[f"linear{i}_w"] = getattr(self, f"linear{i}_w")
            self.w[f"linear{i}_b"] = getattr(self, f"linear{i}_b")
            self.nb_layers += 1

    def my_linear(self, x, weight, bias):
        # return x @ weight.t() + bias.
        #  Although this is the same, they yield different results as here: https://discuss.pytorch.org/t/differences-between-implementations/129237
        return F.linear(x, weight, bias)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)

        residual = x
        x = F.relu(self.my_linear(x, self.linear0_w, self.linear0_b))
        x = self.my_linear(x, self.linear1_w, self.linear1_b)
        x = F.relu(x) + residual
        x = self.my_linear(x, self.linear2_w, self.linear2_b)
        x = F.relu(x) + residual
        x = self.my_linear(x, self.linear3_w, self.linear3_b)
        return x

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
def evaluate(device, model, criterion, val_loader, is_pyquant=False):

    val_loss_running, val_acc_running = 0, 0

    model.eval().cuda() if (device.type == "cuda") else model.eval().cpu()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if is_pyquant:
                outputs = model.forward_pyquant(inputs)
            else:
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
