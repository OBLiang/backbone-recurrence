# -*- coding: utf-8 -*-
# @Time    : 2020/4/5 20:41
# @Author  : DforceL
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable

epoch=1
batch_size=50
LR=0.001
down_mnist=False
train_data=torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=down_mnist,
)
train_loader=Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_data=torchvision.datasets.MNIST(
    root="./mnist",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    # down_mnist=down_mnist,
)
test_loader=Data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)
# print train_data
# print(train_data.data.size())
# print(train_data.targets.size())
# plt.imshow(train_data.data[0].numpy(),cmap="gray")
# plt.title("%i" % train_data.targets[0])
# plt.show()
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output    # return x for visualization
model=CNN()
# print(lenet)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=LR)

# lenet.eval()
counter=0
for epo in range(epoch):
    for img,label in train_loader:
        counter+=1
        # img=Variable(img)
        # label=Variable(label
        out=model(img)
        # print(out[0])
        # print(label[0])
        loss=criterion(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if counter%20==0:
            print("batch:{},loss:{}".format(counter*20,loss.item()))

print("gameover")
