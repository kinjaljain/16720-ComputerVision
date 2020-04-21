import torch
import torch.nn as nn
import torch.utils.data as utils
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32
num_workers = 8
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

train_dir = '../data/oxford-flowers17/train'
train_dataset = ImageFolder(train_dir, transform=preprocess)
train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=True)

model = torchvision.models.squeezenet1_1(pretrained=True)
model.to(device)

last_layer = nn.Conv2d(512, 17, kernel_size=2)
model.classifier = nn.Sequential(nn.Dropout(p=0.4), last_layer, nn.ReLU(inplace=True),
                                 nn.AdaptiveAvgPool2d((2, 2)))
model.type(torch.FloatTensor)
criterion = nn.CrossEntropyLoss().type(torch.FloatTensor)

# only set classifier params to have updated params since we want to fine tuning only that layer
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
num_epochs = 25
train_loss = np.zeros(num_epochs)
train_acc = np.zeros(num_epochs)

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

    train_loss[epoch] = total_loss
    train_acc[epoch] = total_correct / len(train_dataset)
    print('Epoch: {}\t Training Loss: {:.2f}\t Train Accuracy: {:.2f}'.format(epoch + 1, total_loss, train_acc[epoch]))


torch.save(model.state_dict(), "q6_2_model.pkl")
torch.save(optimizer.state_dict(), "q6_2_optim.pkl")

plt.figure('Train Accuracy')
plt.plot(range(num_epochs), train_acc, color='g')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.show()

plt.figure('Train Loss')
plt.plot(range(num_epochs), train_loss, color='g')
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Cross-Entropy Loss vs Epochs")
plt.show()
