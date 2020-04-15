# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 10:44
# @Author  : DforceL
import torch
import torch.nn as nn
import torchvision.transforms as transform
import cv2
import numpy as np
'''
predeal input img
'''
origimg_path="./floder/51.jpg"
img=cv2.imread(origimg_path)
img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
opencv2tensor_transformer=transform.Compose([transform.ToTensor(),])
tensor_img=opencv2tensor_transformer(img)
tensor_img=tensor_img.unsqueeze(0)
print(tensor_img.shape)
# print(tensor_img.shape)
'''
model define
deconvoluational network based on Lenet
'''
class DLenet(nn.Module):
    def __init__(self):
        super(DLenet, self).__init__()
        #conv1
        self.conv1= nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=11,
                      stride=1,
                      padding=1,
                      )#64*3->56*6c
        # self.pool1= nn.MaxPool2d(2,2,return_indices=True)#56*6->28*6
        # self.conv2= nn.Conv2d(in_channels=6,
        #               out_channels=10,
        #               kernel_size=9,
        #               stride=1,
        #               padding=1,
        #               )#22*10
        # self.deconv1= nn.ConvTranspose2d(
        #         in_channels=10,
        #         out_channels=6,
        #         kernel_size=9,
        #         stride=1,
        #         padding=1,
        #     )#28*6
        # self.unpool1=  nn.MaxUnpool2d(2,2)#56*6
        self.deconv2=  nn.ConvTranspose2d(
                in_channels=6,
                out_channels=3,
                kernel_size=11,
                stride=1,
                padding=1,
            )#64*3
    def forward(self,x):
        out=self.conv1(x)
        # out,indice=self.pool1(out)
        # out=self.conv2(out)
        #
        # out=self.deconv1(out)
        # out=self.unpool1(out,indice)
        out=self.deconv2(out)
        return out

dlnet=DLenet()
generate_result=dlnet(tensor_img)
generate_result=torch.transpose(generate_result.squeeze(0),0,2)
numpy_img=generate_result.data.numpy()*255
print(numpy_img.shape)
# cv2.imshow("generate img",numpy_img)
cv2.imwrite("./generator/generator2.jpg",numpy_img)
cv2.waitKey(0)