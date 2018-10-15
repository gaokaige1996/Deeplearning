import os,sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torchvision
import pandas as pd
import torch.nn.functional as F
import numpy as np
import cv2

#load path finger

def image_lable(filename):
	images = []
	label = []
	file = open(filename)
	for line in file.readlines():
		l = line.strip('\n')
		image = l.split('\t')[0]
		lab = int(l.split('\t')[1])
		images.append(image)
		label.append(lab)
	return images, label
filetrain = 'finger_train.txt'
images, label_train = image_lable(filetrain)
filetest = 'finger_valid.txt'
images_test, label_test = image_lable(filetrain)
images_test = images_test[:100]
label_test = label_test[:100]
#hyperparameters
EPOCH = 10
BATCH_SIZE = 2
LR = 0.001
width, height = 320, 320
# convert input images into tensors
traindata = []
testdata = []

for img in images:
	img = cv2.imread(img,0)
	rsimg = cv2.resize(img, (width, height))
	np_img = np.array(rsimg)
	traindata.append(np_img)

tensor_train_x = torch.stack([torch.Tensor(i) for i in traindata]) # transform to torch tensors
tensor_train_x = torch.unsqueeze(tensor_train_x, dim=1).type(torch.FloatTensor)  # add a channel dimension as a singleton dimension (now it's: batch_size,1,320,320)

tensor_train_y = torch.from_numpy(np.array(label_train))


# create dataloader
train_data = Data.TensorDataset(tensor_train_x, tensor_train_y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

#####kg###
testimage = images_test
for img in testimage:
	img = cv2.imread(img,0)
	rsimg = cv2.resize(img, (width, height))
	np_img = np.array(rsimg)
	testdata.append(np_img)

tensor_test_x = torch.stack([torch.Tensor(i) for i in testdata])
tensor_test_x = torch.unsqueeze(tensor_test_x,dim=1).type(torch.FloatTensor)
test_y = torch.from_numpy(np.array(label_test))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 320, 320)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=4,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (4, 320, 320)
            nn.BatchNorm2d(4),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 4x4 空间里向下采样, output shape (4, 160, 160)
        )
        self.conv2 = nn.Sequential(  # input shape (4, 160, 160)
            nn.Conv2d(4, 8, 5, 1, 2),  # output shape (8, 80, 80)
            nn.BatchNorm2d(8),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (8, 80, 80)
        )
        self.conv3 = nn.Sequential(  # input shape (8, 80, 80)
            nn.Conv2d(8, 16, 5, 1, 2),  # output shape (16, 80, 80)
            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (16, 40, 40)
        )
        self.conv4 = nn.Sequential(  # input shape (16, 40, 40)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (16, 40, 40)
            nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 20, 20)
        )
        self.conv5 = nn.Sequential(  # input shape (32, 20, 20)
            nn.Conv2d(32, 64, 5, 1, 2),  # output shape (32, 20, 20)
            nn.BatchNorm2d(64),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 10, 10)
        )
        self.conv6 = nn.Sequential(  # input shape (32, 20, 20)
            nn.Conv2d(64, 128, 5, 1, 2),  # output shape (32, 20, 20)
            nn.BatchNorm2d(128),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (128, 5, 5)
        )

        self.out = nn.Linear(128 * 5 * 5, 2)  # fully connected layer, output 2 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output



# initialize our model
model = CNN()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# training the model
total_step = len(train_loader)
for epoch in range(EPOCH):
    for i, (b_x, b_y) in enumerate(train_loader):
        # Forward pass
        outputs = model(b_x)
        loss = loss_func(outputs, b_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, EPOCH, i + 1, total_step, loss.data[0]))

test_output = model(tensor_test_x)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y.numpy(), 'real number')