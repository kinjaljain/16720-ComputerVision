import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from os.path import join

data_dir = '../python/data_bow/data/'


# def convert_x(data):
#     updated_data = []
#     for x in data:
#         x = x.rstrip()
#         img_path = join(data_dir, x)
#         img = Image.open(img_path)
#         img = np.array(img).astype(np.float32)
#         updated_data.append(img)
#     return updated_data
#
# def convert_y(data):
#     updated_data = []
#     for x in data:
#         x = x.rstrip()
#         updated_data.append(x)
#     return updated_data
#
#
# with open('../python/data_bow/data/train_files.txt', 'r') as f:
#     train_x = f.readlines()
# with open('../python/data_bow/data/train_labels.txt', 'r') as f:
#     train_y = f.readlines()
# with open('../python/data_bow/data/test_files.txt', 'r') as f:
#     test_x = f.readlines()
# with open('../python/data_bow/data/test_labels.txt', 'r') as f:
#     test_y = f.readlines()


# class MyDataset(data.Dataset):
#     def __init__(self, x, y):
#         self.x = torch.from_numpy(convert_x(x)).type(torch.float32)
#         self.y = torch.from_numpy(convert_y(y)).type(torch.LongTensor)
#
#     def __getitem__(self, i):
#         return self.x[i].unsqueeze(0), self.y[i]
#
#     def __len__(self):
#         return len(self.x)

# train_dataset = MyDataset(train_x, train_y)
# train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = MyDataset(test_x, test_y)
# test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


batch_size = 32
num_workers = 8
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


preprocess = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

train_data_folder = '../python/data_bow/train/'
train_dataset = datasets.ImageFolder(root=train_data_folder, transform=preprocess)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_data_folder = '../python/data_bow/test/'
test_dataset = datasets.ImageFolder(root=test_data_folder, transform=preprocess)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MyCNNNetwork(nn.Module):
    def __init__(self, output_size):
        super(MyCNNNetwork, self).__init__()
        self.layer_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.layer_3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=512)
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

model = MyCNNNetwork(36)
model.to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()
num_epochs = 25
train_loss = np.zeros(num_epochs)
train_acc = np.zeros(num_epochs)
val_loss = np.zeros(num_epochs)
val_acc = np.zeros(num_epochs)
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
        y = torch.max(y, 1)[1]
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.max(y_pred, dim=1)[1]
        total_correct += torch.sum(torch.eq(pred, y)).item()

    train_loss[epoch] = total_loss
    train_acc[epoch] = total_correct / len(train_dataset)
    print('Epoch: {}\t Training Loss: {:.2f}\t Train Accuracy: {:.2f}'.format(epoch + 1, total_loss, train_acc[epoch]))

    model.eval()
    val_loss_ = 0.0
    val_correct = 0

    for batch_num, (x, y) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        y = torch.max(y, 1)[1]
        loss = criterion(y_pred, y)
        val_loss_ += loss.item()
        pred = torch.max(y_pred, dim=1)[1]
        val_correct += torch.sum(torch.eq(pred, y)).item()

    val_loss[epoch] = val_loss_
    val_acc[epoch] = val_correct / len(valid_dataset)
    print('Epoch: {}\t Validation Loss: {:.2f}\t Validation Accuracy: {:.2f}'.format(epoch + 1, val_loss_, val_acc[epoch]))

    test_loss_ = 0.0
    test_correct = 0

    for batch_num, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        y = torch.max(y, 1)[1]
        loss = criterion(y_pred, y)
        test_loss_ += loss.item()
        pred = torch.max(y_pred, dim=1)[1]
        test_correct += torch.sum(torch.eq(pred, y)).item()

    test_loss[epoch] = test_loss_
    test_acc[epoch] = test_correct / len(test_dataset)
    print('Epoch: {}\t Test Loss: {:.2f}\t Test Accuracy: {:.2f}'.format(epoch + 1, test_loss_, test_acc[epoch]))
    model.train()

torch.save(model.state_dict(), "q6_1_2_model.pkl")
torch.save(optimizer.state_dict(), "q6_1_2_optim.pkl")

plt.figure('accuracy')
plt.plot(range(num_epochs), train_acc, color='g')
plt.plot(range(num_epochs), test_acc, color='b')
plt.legend(['train acc', 'test acc'])
plt.show()

plt.figure('loss')
plt.plot(range(num_epochs), train_loss, color='g')
plt.plot(range(num_epochs), test_loss, color='b')
plt.legend(['train loss', 'test loss'])
plt.show()
