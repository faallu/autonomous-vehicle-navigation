import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import torch
import torchvision
from torchvision import transforms
from custom_data_set import CustomImageDataset
from model_variants import *
from hyperparameters import *
from model_trainer import train_model


from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

# This is the path where you stored your copy of the nuScenes dataset.
DATAROOT = 'data/sets/nuscenes'
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
nusc_can = NuScenesCanBus(dataroot=DATAROOT)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

custom_dataset = CustomImageDataset(nuscenes, nusc_can, DATAROOT, transform=transform)

data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=training_batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=testing_batch_size, shuffle=True)

net = NeuralNetwork().to(device)
train_model(net, custom_dataset, custom_dataset, "test")



#threshold = 500000
# sample_list = custom_dataset.sample_list
# scene_name = nuscenes.get('scene',sample_list[0]['scene_token'])['name']
# scene_messages = nusc_can.get_messages(scene_name, 'vehicle_monitor')
# times = [m['utime'] for m in scene_messages]
# differences = [abs(times[i] - times[i+1]) for i in range(len(times)-1)]

# for sample in sample_list:
#   minimum = np.inf
#   sample_timestamp = sample['timestamp']
#   scene_name = nuscenes.get('scene',sample['scene_token'])['name']
#   scene_messages = nusc_can.get_messages(scene_name, 'vehicle_monitor')
#   r_message = None

#   for message in scene_messages:
#     message_timestamp = message['utime']
#     time_difference = abs(message_timestamp - sample_timestamp)
#     if time_difference < minimum:
#         minimum = time_difference
#         r_message = message

#   print(minimum)


# examples = iter(data_loader)
# sample, label = next(examples)

# print(label[0])
# plt.imshow(np.moveaxis(sample[0].numpy(),0,-1))
# plt.show()


# current_sample = nuscenes.get('sample', nuscenes.scene[0]['first_sample_token'])
# data_tokens = current_sample['data']
# cam_front_data = nuscenes.get('sample_data', data_tokens['CAM_FRONT'])
# sample = nuscenes.get('sample', nuscenes.scene[0]['first_sample_token'])
# print(sample['data']['CAM_FRONT'])

# img_path = os.path.join(DATAROOT, nuscenes.get('sample_data', sample['data']['CAM_FRONT'])['filename'])
# img = np.asarray(Image.open(img_path))
# img = transform(img)
# img = np.moveaxis(img.numpy(), 0, -1)
# print(img)
# plt.imshow(img)
# plt.show()



# def concat_all_samples(nuscenes):
#   sample_list = []
#   for i in range(len(nuscenes.scene)):
#       current_sample = nuscenes.get('sample', nuscenes.scene[i]['first_sample_token'])
#       while current_sample['next'] != '':
#           sample_list.append(current_sample)
#           current_sample = nuscenes.get('sample', current_sample['next'])
#   return sample_list