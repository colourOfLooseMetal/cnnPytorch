import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import numpy as np
import cv2

class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, num_classes)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))



def test_one_image(im, model):
    im = im.convert("RGB")
    transformImForModel = transforms.Compose([transforms.Resize((128, 128)),
                                              transforms.ToTensor()])
    im = transformImForModel(im)
    # print(im.shape)
    im = im[None, :]
    # print(im.shape)
    # print(im)
    with torch.no_grad():
        images = im.to(device)
        # labels = item['label'].to(device)
        outputs = model(images)
        # print(outputs)#predictions array with score but not sure what max score and min score is to threshold
        _, predicted = torch.max(outputs.data, 1)#strongest score
        # print(predicted)
        return(predicted.item())#0 or 1 for our two labels


num_classes = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device", device)
model = CustomConvNet(num_classes=num_classes).to(device)
# model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load("./fishOrNah.pth"))
model.eval()

im = Image.open("./checkCoinPurse.png")


print("0 fish 1 not fish",test_one_image(im, model))#0 fish 1 not fish
