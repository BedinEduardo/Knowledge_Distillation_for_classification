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
		in_features = self.classifier[0].in_features
		#out_features = self.features[7].out_channels
		self.hidden_size = in_features
		#self.hidden_size = out_features
	
	def forward(self, x):
		
		x = self.features(x)  # First in all networks pass the input in Conv layers
		print(f"first x.shape: {x.shape}\n")
		flattened_conv_output = torch.flatten(x,1)  # the output of conv layers as a flatten tensor
		print(f"flattened_conv_output.shape: {flattened_conv_output.shape}\n")
		conv_feature_maps = x  # this line will be used to Distillate by feature maps
		print(f"conv_feature_maps.shape: {conv_feature_maps.shape}\n")
		x = self.classifier(flattened_conv_output)  # pass the flattened output in the classifier
		print(f"second x.shape: {x.shape}\n")
		flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output,2) # is a vector with dimmensionality adjusted to be used in loss function
		print(f"flattened_conv_output_after_pooling.shape: {flattened_conv_output_after_pooling.shape}\n")

		return x, flattened_conv_output_after_pooling, conv_feature_maps # x is the predicted value (logits) - 
													  # x is the feature map of the last layer
													  # flattened_conv_output_after_pooling is the logits of the conv layers outputs
	

class EfficientNetB0Teacher(nn.Module):
	def __init__(self, num_classes: int, pretrained=True):
		super(EfficientNetB0Teacher, self).__init__()
		
		self.weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT  # best available weights
		self.model = torchvision.models.efficientnet_b0(weights=self.weights)

		# Replace the original classifiers for the current number of classes
		in_features = self.model.classifier[1].in_features
		self.model.classifier[1] = nn.Linear(in_features,num_classes)

		# Save hidden size for KD - EfficientNetB0
		self.hidden_size = in_features # 640 # in_features # should test one by one of the Networks and adjust for each one 
								# This 640 is used in EfficientNetB0 as teacher and LightNN as Student
		#self.add_module = in_features

	def forward(self,x):
		# Step 1: Extract feature maps from LAST Conv block
		print("\nPRINTING TEACHER EFFICIENTNETB0")
		x = self.model.features(x)  # last conv features - pass through the input into conv layers
		print(f"\n1st X Teacher shape: {x.shape}\n")
		
		# Step 2: Flatten for linear classifier
		pooled = self.model.avgpool(x)  # This line is from the Architecture of EfficientNetB0 - and the others....
		print(f"Teacher pooled shape: {pooled.shape}\n")		
		flattened_conv_output = torch.flatten(pooled,1)  # flatten the pooled layer --> prepares features for classifier - like original EfficientNet foward
		print(f"Flattened_conv_output_shape Teacher: {flattened_conv_output.shape}\n")
				
		conv_feature_maps = x  # gets the conv maps from 1st block - model features
		print(f"conv_feature_maps teacher shape: {conv_feature_maps.shape}\n")

		# Step 3. Get logits
		x = self.model.classifier(flattened_conv_output) # pass the flattened output in the classifier.
		print(f"Output X teacher shape: {x.shape}\n")
		
		# Step 4. BUild reduced feature vector for distillation los
		flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output,2)
		print(f"flattened_conv_output_after_pooling teacher shape: {flattened_conv_output_after_pooling.shape}\n")
		
		return x, flattened_conv_output_after_pooling, conv_feature_maps	
		# X is used to distillate knowledge from last layer trough loss functions like Cross-Entropy Loss
		# flattened conv_out_put is used to distillate knowledge from flattened vectors for the output of Conv Layers - or before the classifier
		# conv_feature_maps are used to KD from feature maps of output from conv layers


