# -*- coding: utf-8 -*-
# @Time    : 2020/4/5 12:18
# @Author  : DforceL
import torch
import torch.nn as nn
# 定义VGG网络
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.feature=nn.Sequential(

        )