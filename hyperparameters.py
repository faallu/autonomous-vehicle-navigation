epochs = 1
training_batch_size = 4
testing_batch_size = 4
l1_lambda = 0.0001
l2_lambda = 0.01



import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
