import torch
from torch import nn
from hyperparameters import *

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=0.5)
		self.oh_my_pc = nn.AvgPool2d(5)

		self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
		self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
		self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
		self.conv4 = nn.Conv2d(48, 64, 3)
		self.conv5 = nn.Conv2d(64, 64, 3)
		self.fc1 = nn.Linear(1152, 100)
		self.fc2 = nn.Linear(100, 50)
		self.fc3 = nn.Linear(50, 10)
		self.fc4 = nn.Linear(10, 1)

	def forward(self, x):
		x = self.oh_my_pc(x)
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		x = self.oh_my_pc(self.relu(self.conv5(x)))
		x = torch.flatten(x, 1)
		#print(x.shape)
		x = self.dropout(self.relu(self.fc1(x)))
		x = self.dropout(self.relu(self.fc2(x)))
		x = self.dropout(self.relu(self.fc3(x)))
		x = self.fc4(x)
		return x
	