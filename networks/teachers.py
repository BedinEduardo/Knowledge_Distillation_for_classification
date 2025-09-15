# this code is reponsible to manage the teachers networks
# The firsts models are "own-modeled" models
# But will be needed to add pretrained models before - in the next steps of developments
import torch
import torch.nn as nn
#import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def select_teacher(model:str, num_classes:int):
	if model == "DeepNN":
		teacher = DeepNN(num_classes)
	
	elif model == "EfficientNetB0Teacher":
		teacher = EfficientNetB0Teacher(num_classes)
	
	return teacher


class DeepNN(nn.Module):
	def __init__(self, num_classes: int):  # here in this example we will only use as hyperparameter the num_classes but in further versions can be added more hyperparameters like kernel_size, hidden layers, padding, and so on...
		super(DeepNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3,128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128,64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, padding=1),
			nn.Conv2d(64,64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64,32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.classifier = nn.Sequential(
			nn.Linear(2048,512),
			nn.ReLU(),
			nn.Dropout2d(0.1),
			nn.Linear(512,num_classes)
			)
	
	def forward(self, x):
		x = self.features(x)  # First in all networks pass the input in Conv layers
		#x = torch.flatten(x,1) # This is the original code, but for further distillations is needed to transform in a specific variable to return it to destilation process		
		flattened_conv_output = torch.flatten(x,1)  # the output of conv layers as a flatten tensor
		conv_feature_maps = x  # this line will be used to Distillate by feature maps
		x = self.classifier(flattened_conv_output)  # pass the flattened output in the classifier
		flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output,2) # is a vector with dimmensionality adjusted to be used in loss function

		return x, flattened_conv_output_after_pooling, conv_feature_maps # x is the predicted value (logits) - 
													  # x is the feature map of the last layer
													  # flattened_conv_output_after_pooling is the logits of the conv layers outputs
	

class EfficientNetB0Teacher(nn.Module):
	def __init__(self, num_classes: int, pretrained=True):
		super(EfficientNetB0Teacher, self).__init__()
		# loading pretrained EfficientNetB0
		#self.model = models.efficientnet_b0(pretrained=pretrained)  # Check
		self.weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT  # best available weights
		self.model = torchvision.models.efficientnet_b0(weights=self.weights)

		# Replace the original classifiers for the current number of classes
		in_features = self.model.classifier[1].in_features
		self.model.classifier[1] = nn.Linear(in_features,num_classes)

	def forward(self,x):
		# Step 1: Extract feature maps from LAST Conv block
		# EfficientNet separates features and classifier
		x = self.model.features(x)  # last conv features - pass through the input into conv layers
		
		# Step 2: Flatten for linear classifier
		pooled = self.model.avgpool(x)
		flattened_conv_output = torch.flatten(pooled,1)  # flatten the pooled layer --> prepares features for classifier - like original EfficientNet foward
		conv_feature_maps = x
		
		# Step 3. Get logits
		x = self.model.classifier(flattened_conv_output) # pass the flattened output in the classifier.

		# Step 4. BUild reduced feature vector for distillation los
		flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output,2)

		return x, flattened_conv_output_after_pooling, conv_feature_maps	


