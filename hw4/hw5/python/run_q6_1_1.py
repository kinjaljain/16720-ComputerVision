import numpy as np
import scipy.io
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

# we don't need labels now!
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']


class MyDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).type(torch.float32)
        self.y = torch.from_numpy(y).type(torch.LongTensor)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


batch_size = 32
num_workers = 8
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

train_dataset = MyDataset(train_x, train_y)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = MyDataset(valid_x, valid_y)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MyDataset(test_x, test_y)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MyFCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyFCNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.sigmoid(x)
        x = self.layer_2(x)
        return x


model = MyFCNetwork(1024, 64, 36)
model.to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()
num_epochs = 150
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
        nn.utils.clip_grad_norm(model.parameters(), 0.25)
        optimizer.step()

        total_loss += loss.item()
        pred = torch.max(y_pred, dim=1)[1]
        total_correct += torch.sum(torch.eq(pred, y)).item()

    train_loss[epoch] = total_loss / len(train_loader)
    train_acc[epoch] = total_correct / len(train_dataset)
    print('Epoch: {}\t Training Loss: {:.2f}\t Train Accuracy: {:.2f}'.format(epoch + 1, total_loss[epoch], train_acc[epoch]))

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

    val_loss[epoch] = val_loss_ / len(valid_loader)
    val_acc[epoch] = val_correct / len(valid_dataset)
    print('Epoch: {}\t Validation Loss: {:.2f}\t Validation Accuracy: {:.2f}'.format(epoch + 1, val_loss_[epoch], val_acc[epoch]))

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

    test_loss[epoch] = test_loss_ / len(test_loader)
    test_acc[epoch] = test_correct / len(test_dataset)
    print('Epoch: {}\t Test Loss: {:.2f}\t Test Accuracy: {:.2f}'.format(epoch + 1, test_loss_[epoch], test_acc[epoch]))
    model.train()

torch.save(model.state_dict(), "q6_1_1_model.pkl")
torch.save(optimizer.state_dict(), "q6_1_1_optim.pkl")

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
