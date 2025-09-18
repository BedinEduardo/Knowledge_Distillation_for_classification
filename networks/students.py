# this code is reponsible to manage the Students networks
# The firsts models are "own-modeled" models
# But will be needed to add pretrained models before - in the next steps of developments

import torch
import torch.nn as nn
#import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

class LightNN(nn.Module):
	def __init__(self, num_classes: int, teacher: nn.Module, input_size=(1,3,32,32)): #teacher_hidden_size: int):
		super(LightNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3,16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16,16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		# SHOULD GET THE VALUES OF CONV OUTPUT FROM HERE - But maybe it si static... fro these NN
		teacher.eval() # A run time autodetection version of teacher output.
		with torch.no_grad():
			dummy_imput = torch.zeros(1, *input_size).to(device)   #
			_, flattened_conv_output_after_pooling, conv_maps = teacher(dummy_imput)
			teacher_hidden_size = conv_maps.size(1)  #(1)   # The error is HERE - THIS LINE IS TO BE USED IN CONV FEATURES KD
			teacher_hidden_size_flattened_conv = flattened_conv_output_after_pooling.size(1)
			print(f"teacher_hidden_size: {teacher_hidden_size}")
			print(f"teacher_hidden_size_flattened_conv: {teacher_hidden_size_flattened_conv}")
													  # THE PREVIOUS CODE USED ANOTHER WAY TO GET THE TEACHER HIDDEN LAYERS - 
													  # EACH ONE SHOULD USE A SPECIFIC MODE TO GET THE INFORMATION NEEDS
													  # SEE WHAT THERE WAS BEFORE AND CORRECT THE CODE

		self.regressor = nn.Sequential(		# Convolutional Regressor --> useful to match features maps fo the teacher and the students
			nn.Conv2d(16,teacher_hidden_size, kernel_size=3, padding=1) # kernel_size=3  # Adapted to the size of teacher hidden_layers
		) 
		self.classifier = nn.Sequential(
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(256, num_classes)
		)

		# Store target hidden size
		self.teacher_hidden_size_flattened_conv = teacher_hidden_size_flattened_conv #// 2  # for MobileNetB0 - Check how to do it for others Networks - how to perform it for other NN combinations?
															  # This line is used to get the values of flattened_conv_output_regressor - for EfficientNetB0 is workig, but when use DeepNN it is not working
		self.regressor_teacher = None # lazy int
	
	def forward(self, x):
		
		x = self.features(x)
		print(f"\nFirst Student X shape: {x.shape}\n")
		#x = torch.flatten(x,1)
		flattened_conv_output = torch.flatten(x,1)  # in the student network --> will be used this line before the classfier in cosine 

		# To adapt the flattened_conv_output to the teacher - EfficientNetB0 output shape
		# Build a regressor on first pass - when teacher_hidden_size is known
		if self.regressor_teacher is None:
			if self.teacher_hidden_size_flattened_conv is None:
				raise ValueError("Teacher hidden size must be provideded\n")
			
			self.regressor_teacher = nn.Linear(flattened_conv_output.size(1), self.teacher_hidden_size_flattened_conv).to(x.device)  # Flatened regressor --> linear regressor --> distillation loss is defined on vectors -- CossineEmbeddingLoss, MSELoss
			print(f"teacher_hiddden_size.shape: {self.teacher_hidden_size_flattened_conv}\n")
			print(f"regressor_teacher.shape for flattened_conv_output by student: {self.regressor_teacher}\n")
		
		flattened_conv_output_regressor = self.regressor_teacher(flattened_conv_output)
					
		print(f"flattened_conv_output_regressor.shape: {flattened_conv_output_regressor.shape}\n")
		#print(f"regressor_output original.shape: {regressor_output.shape}\n")
		regressor_output = self.regressor(x) #_teacher(x) # self.regressor(x) # Trainable layer that convert the feature maps of the student to the shape of the teacher.
		print(f"regressor_output student adapted: {regressor_output.shape}\n")
		
		x = self.classifier(flattened_conv_output)
		print(f"Output x student: {x.shape}\n")

		#return x, flattened_conv_output, regressor_output    # x is the logits from the last layers
											# flattened_conv_output is the logits of conv layers outputs
		return x, flattened_conv_output_regressor, regressor_output
		# x is used to KD from outut layer using loss functions like Cross-Entropy Loss
		# flattened_conv_output_regressor is used to KD from flattend vectors of the conv layers output - is used a regressor function to adjust the shapes
		# regressor_output is used to KD from feature maps from the output of the conv layers - is used a regressor function to adjust the shapes