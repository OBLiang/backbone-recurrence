# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 10:44
# @Author  : DforceL
import torch
import torch.nn as nn
import torchvision.transforms as transform
import cv2
import numpy as np
# origimg_path="./floder/50.jpg"
# img=cv2.imread(origimg_path)
# print(img.shape)
# img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("./floder/50.jpg",img)
# cv2.imshow("original_pic",img)
# cv2.waitKey(0)
# print(img.shape)
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
class DLenet_convpart(nn.Module):
    def __init__(self):
        super(DLenet_convpart, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=11,
                      stride=1,
                      padding=1,
                      ),#64*3->56*6
            # nn.MaxPool2d(2,return_indices=True),#56->28
            # nn.Conv2d(in_channels=6,
            #           out_channels=10,
            #           kernel_size=9,
            #           stride=1,
            #           padding=1,
            #           ),#50*10
            # nn.MaxPool2d(2,return_indices=True),#20->10
            # nn.Conv2d(in_channels=10,
            #           out_channels=16,
            #           kernel_size=9,
            #           stride=1,
            #           padding=1,
            # ),#44*16
        )
    def forward(self,x):
        x=self.conv(x)
        # x=self.convtranspose(x)
        return x
class Dlent_deconvpart(nn.Module):
    def __init__(self):
        super(Dlent_deconvpart, self).__init__()
        self.deconv=nn.Sequential(
            # nn.ConvTranspose2d(
            #     in_channels=16,
            #     out_channels=10,
            #     kernel_size=9,
            #     stride=1,
            #     padding=1,
            # ),
            # nn.ConvTranspose2d(
            #     in_channels=10,
            #     out_channels=6,
            #     kernel_size=9,
            #     stride=1,
            #     padding=1,
            # ),
            nn.ConvTranspose2d(
                in_channels=6,
                out_channels=3,
                kernel_size=11,
                stride=1,
                padding=1,
            ),
        )
    def forward(self,x):
        x=self.deconv(x)
        return x

convpart=DLenet_convpart()
deconvpart=Dlent_deconvpart()
conv_result=convpart(tensor_img)
print(conv_result.shape)
deconv_result=deconvpart(conv_result)
deconv_result=deconv_result.squeeze(0)
deconv_result=torch.transpose(deconv_result,0,2)
print(deconv_result.shape)
numpy_img=deconv_result.data.numpy()*255
print(numpy_img.shape)
# numpy_img=cv2.merge([b,g,r])
cv2.imshow("generate img",numpy_img)
cv2.imwrite("./generator/generator2.jpg",numpy_img)
cv2.waitKey(0)