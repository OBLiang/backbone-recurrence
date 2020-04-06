# -*- coding: utf-8 -*-
# @Time    : 2020/4/5 13:41
# @Author  : DforceL
# 两层卷积两层池化交替出现，最后三层全连接层
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False
# 导入mnist dataset
train_data=torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_data=torchvision.datasets.MNIST(
    root="./mnist",
    train=False,
    transform=torchvision.transforms.ToTensor(),
)
test_loader=Data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=False)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2
            ),#input shape 1*28*28->output shape=6*28*28
            nn.MaxPool2d(2,2),#output shape=6*14*14
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
            ),#output shape=16*10*10
            nn.MaxPool2d(2,2),#output shape=16*5*5
        )
        self.classfier=nn.Sequential(
            nn.Linear(16*5*5,10),
        )
    def forward(self,x):
        x=self.feature(x)
        x=x.view(x.size(0),-1)
        output=self.classfier(x)
        return output
lenet=LeNet()


lossfun=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(lenet.parameters(),lr=LR)

counter=0
for epo in range(EPOCH):
    for img,label in train_loader:
        counter+=1
        out=lenet(img)
        loss=lossfun(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(counter%20==0):
            print("batch:{},loss:{}".format(counter*20,loss.item()))
