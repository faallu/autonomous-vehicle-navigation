import torch
from torch import nn
import torch.optim as optim
from enum import Enum, auto
import json
from hyperparameters import *
from model_variants import *
import matplotlib.pyplot as plt
import numpy as np
import os

def train_model(net, training_data, test_data, output_name, l1_reg = False, l2_reg = False):
	train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=training_batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=training_batch_size, shuffle=True)
	criterion = nn.MSELoss(reduction='mean').to(device)
	optimizer = optim.Adam(net.parameters())

	class Phases(Enum):
		TRAINING = auto()
		VALIDATION = auto()

	phases = [Phases.TRAINING, Phases.VALIDATION]

	datasets = {Phases.TRAINING : (train_loader, training_data), Phases.VALIDATION : (test_loader, test_data)}

	train_statistics = []
	validation_statistics = []
	torch.backends.cudnn.benchmark = True
	for epoch in range(epochs):  # loop over the dataset multiple times
		statistics = {Phases.TRAINING : (0, 0), Phases.VALIDATION : (0, 0)}
		for phase in phases:
			running_loss = 0.0
			running_corrects = 0.0
			loader, dataset = datasets[phase]

			for i, data in enumerate(loader, 0):
			# get the inputs; data is a list of [inputs, labels]
				torch.set_grad_enabled(True if phase == Phases.TRAINING else False)
				inputs, labels = data
				inputs = inputs.to(device)
				labels = labels.to(device).to(torch.float32).unsqueeze(1)
				# zero the parameter gradients
				for param in net.parameters():
					param.grad = None

				outputs = None
				loss = None
				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				#print(loss)
				# print('label:',labels)
				# print('output:',outputs)

				if l1_reg:
					l1_reg_value = torch.tensor(0.).to(device)
					for param in net.parameters():
						l1_reg_value +=  torch.sum(param.abs())
					loss += l1_lambda * l1_reg_value
				

				if l2_reg:
					l2_reg_value = torch.tensor(0.).to(device)
					for param in net.parameters():
						l2_reg_value += torch.linalg.norm(param)
					loss += l2_lambda * l2_reg_value

				if phase == Phases.TRAINING:
					loss.backward()
					optimizer.step()
				
				
				predictions = outputs.data 

				running_loss += loss.item() * loader.batch_size
				running_corrects += torch.sum(torch.isclose(predictions, labels, atol=1e-3, rtol=0)).item()

				statistics[phase] = (running_loss/len(dataset), running_corrects/len(dataset))
			
		epoch_training_loss, epoch_training_accuracy = statistics[Phases.TRAINING]
		epoch_validation_loss, epoch_validation_accuracy = statistics[Phases.VALIDATION]

		train_statistics.append(statistics[Phases.TRAINING])
		validation_statistics.append(statistics[Phases.VALIDATION])
		
		print(f"Epoch: {epoch+1}, Train Loss: {epoch_training_loss:.3f}, Train Accuracy: {epoch_training_accuracy:.3f}, Validation Loss: {epoch_validation_loss:.3f}, Validation Accuracy: {epoch_validation_accuracy:.3f}")

	#unpack the data
	train_loss_stats = [x[0] for x in train_statistics]
	train_accuracy_stats = [x[1] for x in train_statistics]
	validation_loss_stats = [x[0] for x in validation_statistics]
	validation_accuracy_stats = [x[1] for x in validation_statistics]

	#format for json
	stats_dict = {
		"training loss" : train_loss_stats,
		"training accuracy" : train_accuracy_stats,
		"validation loss" : validation_loss_stats,
		"validation accuracy" : validation_accuracy_stats
	}

	torch.save({
		'epoch': epoch,
		'model_state_dict': net.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
	}, 'test.pt')

	with open(f"results/{output_name}.json", "w") as outfile:
		json.dump(stats_dict, outfile, indent=4)

	print(f"Finished Training {output_name}")


