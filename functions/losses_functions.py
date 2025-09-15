# A code to select the loss function according the value seted by the researcher

import torch
import torch.nn as nn

def select_loss(loss_function):
	# A code to select the loss function according the user choice
	if loss_function == 1:
		loss_fn = nn.CrossEntropyLoss()
	
	elif loss_function == 2:
		loss_fn = nn.CosineEmbeddingLoss()
	
	elif loss_function == 3:
		loss_fn = nn.MSELoss()
	
	else:
		raise ValueError(f"Invalid loss function value: {loss_function}. Expected 1 or 2")
	
	return loss_fn
	
# add more accordling the need of losses_function that are being increasing