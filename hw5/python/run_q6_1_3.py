import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

batch_size = 32
num_workers = 8
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

transform_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_2 = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_1)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MyCNNNetwork(nn.Module):
    def __init__(self, output_size):
        super(MyCNNNetwork, self).__init__()
        self.layer_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.layer_3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(256, 128)
        self.dp1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.layer_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.layer_3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dp1(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

model = MyCNNNetwork(10)
model.to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()
num_epochs = 25
train_loss = np.zeros(num_epochs)
train_acc = np.zeros(num_epochs)
test_loss = np.zeros(num_epochs)
test_acc = np.zeros(num_epochs)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    for batch_num, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.max(y_pred, dim=1)[1]
        total_correct += torch.sum(torch.eq(pred, y)).item()

    train_loss[epoch] = total_loss / len(train_loader)
    train_acc[epoch] = total_correct / len(train_dataset)
    print('Epoch: {}\t Training Loss: {:.2f}\t Train Accuracy: {:.2f}'.format(epoch + 1, train_loss[epoch], train_acc[epoch]))

    model.eval()
    test_loss_ = 0.0
    test_correct_ = 0

    for batch_num, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        test_loss_ += loss.item()
        pred = torch.max(y_pred, dim=1)[1]
        test_correct_ += torch.sum(torch.eq(pred, y)).item()

    test_loss[epoch] = test_loss_ / len(test_loader)
    test_acc[epoch] = test_correct_ / len(test_dataset)
    print('Epoch: {}\t Test Loss: {:.2f}\t Test Accuracy: {:.2f}'.format(epoch + 1, test_loss[epoch], test_acc[epoch]))
    model.train()

torch.save(model.state_dict(), "q6_1_3_model.pkl")
torch.save(optimizer.state_dict(), "q6_1_3_optim.pkl")

plt.figure('Accuracy')
plt.plot(range(num_epochs), train_acc, color='g')
plt.legend(['Train Accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.show()

plt.figure('Accuracy')
plt.plot(range(num_epochs), train_acc, color='g')
plt.plot(range(num_epochs), test_acc, color='b')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.show()

plt.figure('Cross-Entropy Loss')
plt.plot(range(num_epochs), train_loss, color='g')
plt.legend(['Train Loss'])
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Cross-Entropy Loss vs Epochs")
plt.show()

plt.figure('Cross-Entropy Loss')
plt.plot(range(num_epochs), train_loss, color='g')
plt.plot(range(num_epochs), test_loss, color='b')
plt.legend(['Train Loss', 'Test Loss'])
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Cross-Entropy Loss vs Epochs")
plt.show()