# -*- coding: utf-8 -*-
# @Time    : 2020/4/5 13:41
# @Author  : DforceL
# 两层卷积两层池化交替出现，最后三层全连接层
import torch
import torchvision
import torch.nn as nn
EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature=nn.Sequential(
            nn.Conv2d(1,6,3,1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.MaxPool2d(2,2),
        )
        self.classfier=nn.Sequential(
            nn.Linear(400,200),
            nn.Linear(120,84),
            nn.Linear(84,10),
        )
    def forward(self,x):
        x=self.feature(x)
        x=self.classfier(x)

lenet=LeNet()
print(lenet)