# A code to select a activation function according by the value seted by the reseacher

import torch
import torch.nn as nn
import torch.optim as optim

def select_optimizer(optimizer_func, model, learning_rate):

	if optimizer_func == 1:
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # adapt this part of the code to be loaded by a function --> receive the value and return the optimizer and the loss function
	
	return optimizer

	# add more accordling the need of optimizers that are being increasing