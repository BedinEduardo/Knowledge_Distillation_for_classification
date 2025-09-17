# This is the code to run the training, validation and test steps
# In this first version, only evaluate, but before it is needed to add testing step.

import torch
import torch.nn as nn
import torch.optim as optim

import time

from functions.losses_functions import select_loss
from functions.optimizer_functions import select_optimizer

from typing import Dict, List
from tqdm.auto import tqdm

# tranform each class --> for first training or destilation in a class each one

class TrainTestBeforeDistil():
	"""
	A class to train techer and student network to destile knowledge by Loss Function in the output layer
	This class is to be used in Classification KD tasks
	"""
	def train(model: torch.nn.Module,
		   train_loader: torch.utils.data.DataLoader,
		   learning_rate:float,
		   loss_func: int,
		   optimizer_func: int,
		   device: torch.device):  # can be added more hyperparameters before
		# this function will be used for the first training step for teacher and student --> before other function to destillate the knowledge

		loss_fn = select_loss(loss_func)

		optimizer = select_optimizer(optimizer_func, model, learning_rate)
		
		model.train()
		
		train_loss, train_acc = 0, 0
		# iterate trough the DataLoader
		for batch, (inputs, labels) in enumerate(train_loader):
			inputs, labels = inputs.to(device), labels.to(device)

			# 1. Forward pass
			output, _, _ = model(inputs)

			# 2. Calculate and accumulate the loss
			loss = loss_fn(output, labels)
			train_loss += loss.item()

			# 3. Optimizer zero grad
			optimizer.zero_grad()

			# 4. Loss backward
			loss.backward()
			
			# 5. optimizer.step()
			optimizer.step()

			# calculate and accumulate accuracy metric across all batches
			output = torch.argmax(torch.softmax(output, dim=1),dim=1)
			train_acc += (output == labels).sum().item()/len(labels)  # for classification
		
		# Adjust metrics to get average loss and accuracy per batch
		train_loss = train_loss / len(train_loader)
		train_acc = train_acc / len(train_loader)

		return train_loss, train_acc
		

	def test(model: torch.nn.Module,
		  test_loader: torch.utils.data.DataLoader,
		  loss_func: int,
		  loss_func_out_layer: int,
		  loss_func_soft_label: int,
		  loss_func_hidden_layers:int,
		  KD:int,
		  device: torch.device):
		# this function will be used to the first training steep for teacher and student --> before other function to destillate the knowledge
		# It will be necessary to adapt the code for further versions to perform cross-validation
		#model.to(device)  # check
		model.eval()

		if loss_func is not None:
			loss_fn = select_loss(loss_func)

		elif loss_func is None:
			if KD == 1:
				loss_fn = select_loss(loss_func_out_layer)
			elif KD == 2:
				loss_fn = select_loss(loss_func_out_layer)  # in further versions adapt to use different loss functions
			elif KD == 3:
				loss_fn = select_loss(loss_func_out_layer)

		test_loss, test_acc = 0, 0

		with torch.inference_mode():
			# Loop through DataLoader batches
			for batch, (input, label) in enumerate(test_loader):
				input, label = input.to(device), label.to(device)

				# 1. Forward pass
				output, _, _ = model(input)  # get the logits # SHOULD REMEMBER FOR TRAINING BEFOR KD AND TEST STEPS, ONLY NEEDS THE X VALUE

				# 2. Calculate the loss
				loss = loss_fn(output, label)

				test_loss += loss.item()

				# 3. Calculate and acumulate the accuracy
				output = output.argmax(dim=1)  # transform teh logits into absolute values --> class prediction
				test_acc += ((output == label).sum().item()/len(output))
		
		return test_loss, test_acc

class TrainTestLoops():
	"""
	A Class to select the loops that will iterate over the data according the distillaton step, and according the distillation techinique selected
	"""
	
	def train_loop(teacher: torch.nn.Module, 
				student: torch.nn.Module,
				train_loader: torch.utils.data.DataLoader, 
				test_loader: torch.utils.data.DataLoader, 
				epochs: int,
				learning_rate:float, 
				loss_func_train: int,
				loss_func_out_layer:int,
				loss_func_soft_label:int,
				loss_func_hidden_layers: int,
				optimizer_func_train: int,
				optimizer_func_out_layer:int,
				optimizer_func_soft_label:int,
				optimizer_func_hidden_layers:int,
				KD:int,
				T:float,
				soft_target_loss_weight:float,
				ce_loss_weight_out_layer:float,
				ce_loss_weight_soft_label:float,
				hidden_rep_loss_weight:float,
				ce_loss_weight_hidden_layers:int,
				feature_map_weight:int,
				step: int,
				device: torch.device) ->Dict[str, List[float]]:

		""""
		Train and test a PyTorch Model

		Passes a target PyTorch models through train() and test() steps
		functions for a number of epochs, training and testing the model
		in then same epoch loop.

		Calculates, prints, and store evaluation metrics troughout.

		"""
		def metric_function(epoch, train_loss, train_acc, test_loss, test_acc):
			results = {"train_loss": [],
			 "train_acc": [],
			 "test_loss": [],
			 "test_acc": []
			 }
			print(f"Epoch: {epoch+1} | "
					f"Train_loss: {train_loss:.4f} | "
					f"Train_acc: {train_acc:.4f} | "
					f"Test_loss: {test_loss:.4f} | "
					f"Test_Acc: {test_acc:.4f} \n"
				)
			
			# update results dictionary
			results["train_loss"].append(train_loss)
			results["train_acc"].append(train_acc)
			results["train_loss"].append(test_loss)
			results["test_acc"].append(test_acc)

			return results


		# Build a enpty results dictinonary
		results = {"train_loss": [],
			 "train_acc": [],
			 "test_loss": [],
			 "test_acc": []
			 }
		
		# Loop through training and testing steps for a number of epochs
		if step == 1: # THIS MEANS THAT IS TRAINING BEFORE TO KD - TRAIN THE TEACHER AND TRAIN THE STUDENT FIRST
			print("\n[IN_PROGRESS: ]Starting fist Training the Teacher and Student Networks\n")									
			if teacher is not None:
				print("\n[STEP: ]TRAINING THE TEACHER FIRST \n")
				model = teacher
			elif student is not None:
				print("\n[STEP: ]TRAINING THE STUDENT NETWORK \n")
				model =  student

			for epoch in tqdm(range(epochs)):  # tqdm is a libray to build a progress bar
				print("\n[IN_PROGRESS: ]Training Networks before to start the distillation process")
				train_loss, train_acc = TrainTestBeforeDistil.train(model=model,
														train_loader=train_loader,
														loss_func=loss_func_train,
														learning_rate=learning_rate,
														optimizer_func = optimizer_func_train,
														device=device
														)
				
				test_loss, test_acc = TrainTestBeforeDistil.test(model=model,
													loss_func=loss_func_train,
													loss_func_out_layer=loss_func_out_layer,
													loss_func_soft_label=None,
													loss_func_hidden_layers=None,
													test_loader=test_loader,
													KD=None,
													device=device)
				
				results = metric_function(epoch, train_loss, train_acc, test_loss, test_acc)
		


		elif step == 2: # THIS MEANS THAT IS IN KD STEP - DISTILLATION LOSS ==> Output Layer
			for epoch in tqdm(range(epochs)):
				
				if KD == 1:  # This means that will be KD by output layer
					print("\n[IN_PROGRESS: ]Peforming the Distillation Step in Output Layer using Loss Function\n")
					train_loss, train_acc = TrainKD.train_kd_output_layer(teacher=teacher,  # This part perform the training step using teacher --> student
															student=student,
															train_loader=train_loader,
															epochs=epochs,
															learning_rate=learning_rate,														  
															loss_func_out_layer=loss_func_out_layer,
															loss_func_soft_label=None,
															loss_func_hidden_layers=None,
															optimizer_func_out_layer=optimizer_func_out_layer,
															optimizer_func_soft_label=None,
															optimizer_func_hidden_layers=None,
															KD=KD, # this means that is a KD by the output layer
															T=T,
															soft_target_loss_weight=soft_target_loss_weight,  # it is passed all the parameter bellow and in TraiKD step it is separated by KD task
															ce_loss_weight_out_layer=ce_loss_weight_out_layer,
															ce_loss_weight_soft_label=None,
															hidden_rep_loss_weight=None,
															ce_loss_weight_hidden_layers=None,
															feature_map_weight=None,
															device=device)
					
					test_loss, test_acc = TrainTestBeforeDistil.test(model=student,  # test only for student NN
														loss_func=None,
														loss_func_out_layer=loss_func_out_layer,
														loss_func_soft_label=None,
														loss_func_hidden_layers=None,
														test_loader=test_loader,
														KD=KD,
														device=device)
					
					results = metric_function(epoch, train_loss, train_acc, test_loss, test_acc)

				elif KD == 2:
					print("\n[IN_PROGRESS: ]Peforming the Distillation Step in Soft Labels opf Flattened Vectors from output of Conv Layers\n")
					train_loss, train_acc = TrainKD.train_kd_output_layer(teacher=teacher,  # This part perform the training step using teacher --> student
															student=student,
															train_loader=train_loader,
															epochs=epochs,
															learning_rate=learning_rate,														  
															loss_func_out_layer=loss_func_out_layer,
															loss_func_soft_label=loss_func_soft_label,
															loss_func_hidden_layers=None,
															optimizer_func_out_layer=None,
															optimizer_func_soft_label=optimizer_func_soft_label,
															optimizer_func_hidden_layers=None,
															KD=KD, # this means that is a KD by the output layer
															T=T,
															soft_target_loss_weight=soft_target_loss_weight,  # it is passed all the parameter bellow and in TraiKD step it is separated by KD task
															ce_loss_weight_out_layer=None,
															ce_loss_weight_soft_label=ce_loss_weight_soft_label,
															hidden_rep_loss_weight=hidden_rep_loss_weight,
															ce_loss_weight_hidden_layers=None,
															feature_map_weight=None,
															device=device)
					
					test_loss, test_acc = TrainTestBeforeDistil.test(model=student,  # test only for student NN
														loss_func=None,
														loss_func_out_layer=loss_func_out_layer,
														loss_func_soft_label=loss_func_soft_label,
														loss_func_hidden_layers=None,
														test_loader=test_loader,
														KD=KD,
														device=device)
					results = metric_function(epoch, train_loss, train_acc, test_loss, test_acc)

				elif KD ==3:
					print("\n[IN_PROGRESS: ]Performing the Distillation Step from output of Conv Layers")
					train_loss, train_acc = TrainKD.train_kd_output_layer(teacher=teacher,  # This part perform the training step using teacher --> student
															student=student,
															train_loader=train_loader,
															epochs=epochs,
															learning_rate=learning_rate,														  
															loss_func_out_layer=loss_func_out_layer,
															loss_func_soft_label=None,
															loss_func_hidden_layers=loss_func_hidden_layers,
															optimizer_func_out_layer=None,
															optimizer_func_soft_label=None,
															optimizer_func_hidden_layers=optimizer_func_hidden_layers,
															KD=KD, # this means that is a KD by the output layer
															T=T,
															soft_target_loss_weight=None,  # it is passed all the parameter bellow and in TraiKD step it is separated by KD task
															ce_loss_weight_out_layer=None,
															ce_loss_weight_soft_label=None,
															hidden_rep_loss_weight=None,
															ce_loss_weight_hidden_layers=ce_loss_weight_hidden_layers,
															feature_map_weight=ce_loss_weight_hidden_layers,
															device=device)
					
					test_loss, test_acc = TrainTestBeforeDistil.test(model=student,  # test only for student NN
														loss_func=None,
														loss_func_out_layer=loss_func_out_layer,
														loss_func_soft_label=None,
														loss_func_hidden_layers=loss_func_hidden_layers,
														test_loader=test_loader,
														KD=KD,
														device=device)
					results = metric_function(epoch, train_loss, train_acc, test_loss, test_acc)



		return results

# A class to distile knowledge with Loss function in the output layer
class TrainKD():
	"""
	A class to train teacher and student network to destile knowledge by Loss Function in the output layer
	This class is to be used in Classification KD tasks
	"""
	def train_kd_output_layer(teacher:torch.nn.Module,
						   student:torch.nn.Module,
						   train_loader:torch.utils.data.DataLoader,
						   epochs:int,
						   learning_rate:float,
						   loss_func_out_layer:int,
						   loss_func_soft_label:int,
						   loss_func_hidden_layers:int,
						   optimizer_func_out_layer:int,
						   optimizer_func_soft_label:int,
						   optimizer_func_hidden_layers:int,
						   KD: int,
						   T: float,
						   soft_target_loss_weight:float,
						   ce_loss_weight_out_layer:float,
						   ce_loss_weight_soft_label:float,
						   hidden_rep_loss_weight:float,
						   ce_loss_weight_hidden_layers:float,
						   feature_map_weight: float,
						   device:torch.device) ->Dict[str, List[float]]:
		
		if KD == 1:  # this means KD by output layer usinf loss function
			# Defining loss function and optimizer
			loss_fn_out = select_loss(loss_func_out_layer)  # A Loss function used to calculate the loss in KD in output layer
			print(f"loss_fn_out: {loss_fn_out}")
			optimizer_out = select_optimizer(optimizer_func=optimizer_func_out_layer,
									model=student,
									learning_rate=learning_rate)  # A Optimizer function used to perform the KD in output layer
		
		elif KD == 2:  # This means KD by soft labels
			loss_fn_out = select_loss(loss_func_out_layer)
			print(f"loss_fn_out: {loss_fn_out}")
			loss_fn_soft= select_loss(loss_func_soft_label)  # A Loss function used to calculate the loss in KD using soft labels
			print(f"loss_fn_soft: {loss_fn_soft}")
			optimzer_soft = select_optimizer(optimizer_func=optimizer_func_soft_label,
										  model=student,
										  learning_rate=learning_rate)		# A Optimizer function used to perform the KD in soft labels

		elif KD == 3:
			loss_fn_out = select_loss(loss_func_out_layer)	
			print(f"loss_fn_out: {loss_fn_out}")
			loss_fn_hidden = select_loss(loss_func_hidden_layers)
			optimizer_func_hidden = select_optimizer(optimizer_func=optimizer_func_hidden_layers,
											model=student,
											learning_rate=learning_rate)

		teacher.eval()  # teacher will be in evaluation mode
		student.train()  # Student will keep to be in the training mode.
					
		# ADPAT THE CODE ABOVE TO THE DANIEL BOURKE FORMAT
		train_loss, train_acc = 0, 0 

		# NOW ADAPT THE CODE TO TUN SEPARATELY THE DISTILLATION PROCESS
		for batch, (input, label) in enumerate(train_loader):
			# send the data to the target device
			input, label = input.to(device), label.to(device)

			# 1. Perform the forward pass
			# 1.a Teacher logits
			# WILL BE NEEDED TO ADAPT TO CODE TO MAKE IT TO UNDERSTAND IN EACH DISTILATION STEP IT IS - IF IT IS OUTPUTLAYERE - COSSINE DISTILLATION - OR FEATURE DISTILATION
			with torch.no_grad():
				teacher_logits, teacher_hidden_representation, conv_feature_maps_teacher = teacher(input)  # We need the logits from the conv layers of the teacher
																				# teacher_hidden_representation is the flattened_conv_output_after_pooling 
																				# # in distillation by loss function in last layer will use flattened_conv_output_after_pooling
			#1.b Student logits
			student_logits, student_hidden_representation, regressor_student_output = student(input)  # INSERT A: if student_hidden_representation is not None: to execute different tasks
																			# in distillation by loss function in last layer will use flattened_conv_output_after_pooling
																			# student_hidden_representation is the flattened_conv_output of the student model

			if KD == 1: # This means that it is distillating in output layer
				soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
				soft_probs = nn.functional.log_softmax(student_logits / T, dim=-1)

			# print(f"Type soft_targets: {type(soft_targets), soft_targets} \n")
			# print(f"Type soft_probs: {type(soft_probs), soft_probs} \n")
			
			# print(f"Soft targets shape: {soft_targets.shape}")
			# print(f"Soft probs shape: {soft_probs.shape}")
			# time.sleep(1)

			# 2. Calculate and accumulate the loss
			# for KD there are some additional steps
			#soft_targets_loss = torch.sum(soft_targets * ((soft_targets.log() - soft_probs.size()[0]) * (T*T)))
				soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_probs)) / soft_probs.size()[0] * (T*T)
						
			# Now calculate the true label loss
				label_loss = loss_fn_out(student_logits, label)  # calculate the loss to KD in output layer

			# print(f"Type soft_targets_loss: {type(soft_targets_loss), soft_targets_loss} \n")
			# print(f"Type label_loss: {type(label_loss), label_loss} \n")

			# print(f"soft_targets_loss shape: {soft_targets_loss.shape}")
			# print(f"label_loss shape: {label_loss.shape}")
			# time.sleep(1)
			
			# #ce_loss_weight = float(ce_loss_weight)
			# print("tyep soft_target_loss_weight:", soft_target_loss_weight, type(soft_target_loss_weight))
			# print("type ce_loss_weight:", ce_loss_weight, type(ce_loss_weight))
				new_ce_loss_weight = 1- soft_target_loss_weight
			# print("type new_ce_loss_weight:", new_ce_loss_weight, type(new_ce_loss_weight))
			
			# Now the total loss
				loss = (soft_target_loss_weight * soft_targets_loss) + (new_ce_loss_weight * label_loss)  # HERE IS OCCURING A ERROR

				# 3. Optimizer Zero Grad
				optimizer_out.zero_grad()

				# 4. Loss Backward()
				loss.backward()

				# 5. Optimizer step
				optimizer_out.step()

				train_loss += loss.item()  # itemize each step

			elif KD == 2: # this means that the code is disllating in Soft Labels step --> First in Cosine Minimization Loss Function
				# Calculate the Loss by the hidden layers --> target is a vector of ones.
				hidden_rep_loss = loss_fn_soft(student_hidden_representation, teacher_hidden_representation, target=torch.ones(input.size(0)).to(device))
				# Calculate the true label loss
				label_loss = loss_fn_out(student_logits, label)  # update this in the parameter code  # Use the same Cross-Entropy Loss -> Study new options in further
				
				new_ce_loss_weight = 1- hidden_rep_loss_weight
				# weighted sum of the two losses
				loss = hidden_rep_loss_weight * hidden_rep_loss + new_ce_loss_weight * label_loss

				# 3. Optimizer Step
				optimzer_soft.zero_grad()
				# 4. Loss backwars
				loss.backward()

				# Optimizer step
				optimzer_soft.step()

				train_loss += loss.item()  # itemize each step
			
			elif KD == 3:
				hidden_rep_loss = loss_fn_hidden(regressor_student_output, conv_feature_maps_teacher)
				label_loss = loss_fn_out(student_logits, label)
				
				new_ce_loss_weight_hidden_layers = 1 - feature_map_weight

				loss = feature_map_weight * hidden_rep_loss + new_ce_loss_weight_hidden_layers * label_loss

				# 3. Optimizer Step
				optimizer_func_hidden.zero_grad()

				# 4. Loss backward
				loss.backward()

				# 5. Optmizer Step
				optimizer_func_hidden.step()

				train_loss += loss.item()

			# train_loss += loss.item()  # itemize each step

			# # 3. Optimizer Zero Grad
			# optimizer.zero_grad()

			# # 4. Loss Backward
			# loss.backward()

			# # 5. Optimizer step
			# optimizer.step()

			# Calculate and accuracy metric across all batches
			y_pred_class = torch.argmax(torch.softmax(student_logits, dim=1), dim=1)
			train_acc += (y_pred_class == label).sum().item()/len(student_logits)
		
		# Adjust metrics to get average loss and accuracy per batch
		train_loss = train_loss / len(train_loader)
		train_acc = train_acc / len(train_loader)
		return train_loss, train_acc
			
			#print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")