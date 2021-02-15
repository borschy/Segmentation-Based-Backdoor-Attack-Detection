from torch import nn
import torch.nn.functional as F
import numpy as np


# this isn't finished, 
# need to study it more carefully to find out how it works
# and modify accordingly
# because i think this might be for segmentation,
# not classification 
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*10*10,1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024,20)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1,512*10*10)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)
        return x


#     Model 1
#     _________________________________________________________________
#     Layer (type)                 Output Shape              Param #   
#     =================================================================
#     conv2d_25 (Conv2D)           (None, 254, 254, 32)      896       
#     _________________________________________________________________
#     activation_22 (Activation)   (None, 254, 254, 32)      0         
#     _________________________________________________________________
#     max_pooling2d_19 (MaxPooling (None, 127, 127, 32)      0         
#     _________________________________________________________________
#     conv2d_26 (Conv2D)           (None, 125, 125, 32)      9248      
#     _________________________________________________________________
#     activation_23 (Activation)   (None, 125, 125, 32)      0         
#     _________________________________________________________________
#     max_pooling2d_20 (MaxPooling (None, 62, 62, 32)        0         
#     _________________________________________________________________
#     conv2d_27 (Conv2D)           (None, 60, 60, 64)        18496     
#     _________________________________________________________________
#     activation_24 (Activation)   (None, 60, 60, 64)        0         
#     _________________________________________________________________
#     max_pooling2d_21 (MaxPooling (None, 30, 30, 64)        0         
#     _________________________________________________________________
#     flatten_11 (Flatten)         (None, 57600)             0         
#     _________________________________________________________________
#     dense_14 (Dense)             (None, 64)                3686464   
#     _________________________________________________________________
#     activation_25 (Activation)   (None, 64)                0         
#     _________________________________________________________________
#     dropout_11 (Dropout)         (None, 64)                0         
#     _________________________________________________________________
#     dense_15 (Dense)             (None, 4)                 260       
#     _________________________________________________________________
#     activation_26 (Activation)   (None, 4)                 0  