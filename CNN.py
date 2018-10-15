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


#hyperparameters
EPOCH = 10
BATCH_SIZE = 2
LR = 0.001
width, height = 320, 320
# convert input images into tensors
traindata = []
testdata = []

#images = ['image1.png','image2.png','image3.png','image4.png','image5.png','image6.png','image7.png','image8.png']
for img in images:
	img = cv2.imread(img,0)
	rsimg = cv2.resize(img, (width, height))
	np_img = np.array(rsimg)
	traindata.append(np_img)

tensor_train_x = torch.stack([torch.Tensor(i) for i in traindata]) # transform to torch tensors
tensor_train_x = torch.unsqueeze(tensor_train_x, dim=1).type(torch.FloatTensor)  # add a channel dimension as a singleton dimension (now it's: batch_size,1,320,320)

tensor_train_y = torch.from_numpy(np.array(label_train))
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
#######################
# create dataloader
train_data = Data.TensorDataset(tensor_train_x, tensor_train_y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# define our 169 layer model with first layer: CNN and 168 fully connected layers, followed by the output softmax layer
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(         # input shape (1, 320, 320)
			nn.Conv2d(
				in_channels=1,
				out_channels=4,
				kernel_size=3,
				stride=1,
				padding=1),					# output shape (4, 320, 320)
			nn.BatchNorm2d(4),
			nn.ReLU(),                     
			nn.MaxPool2d(8),	# output shape (4, 40, 40)
		)
		#self.conv2 = nn.Sequential(

		#)
		self.fcs = []
		self.bns = []
		for i in range(168):
			input_size = 4*40*40 if i == 0 else 128
			fc = nn.Linear(input_size,128)
			setattr(self,'fc%i'%i, fc)
			self.fcs.append(fc)

			bn = nn.BatchNorm1d(128,momentum=0.5)
			setattr(self,'bn%i'%i, bn)
			self.bns.append(bn)
			
		self.out = nn.Linear(128, 2)        # input shape (128, 2)
		
	def forward(self, x):
		x = self.conv1(x)
		x = x.view(x.size(0), -1) 	# flatten the output of classifier2 to (batch_size, 128)
		for i in range(168):
			x = self.fcs[i](x) 	# fully connected
			x = F.tanh(x)		# activation
			x = self.bns[i](x)	# batch normalization

		x = self.out(x)
		
		return x

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
		
		if (i+1) % 2 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, EPOCH, i+1, total_step, loss.data[0]))


test_output = model(tensor_test_x)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y.numpy(), 'real number')