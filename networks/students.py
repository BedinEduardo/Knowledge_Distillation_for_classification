# this code is reponsible to manage the Students networks
# The firsts models are "own-modeled" models
# But will be needed to add pretrained models before - in the next steps of developments

import torch
import torch.nn as nn
#import torch.optim as optim

class LightNN(nn.Module):
	def __init__(self, num_classes: int):
		super(LightNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3,16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16,16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.regressor = nn.Sequential(
			nn.Conv2d(16,32, kernel_size=3, padding=1)
		)
		self.classifier = nn.Sequential(
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(256, num_classes)
		)
	
	def forward(self, x):
		x = self.features(x)
		#x = torch.flatten(x,1)
		flattened_conv_output = torch.flatten(x,1)  # in the student network --> will be used this line before the classfier in cosine 
		regressor_output = self.regressor(x) # Trainable layer that convert the feature maps of the student to the shape of the teacher.
		x = self.classifier(flattened_conv_output)

		return x, flattened_conv_output, regressor_output    # x is the logits from the last layers
											# flattened_conv_output is the logits of conv layers outputs	