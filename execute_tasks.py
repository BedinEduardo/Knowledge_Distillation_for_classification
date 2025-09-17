# A code to be used to load the tasks defined by the users

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

import time

from datasets.load_dataset import Load_Standard_Datasets  # starting importing CIFAR10 Datasets but -- the code will be adapted to perform other datasets too
from networks.teachers import select_teacher
from networks.students import LightNN
from train_test_step import TrainTestBeforeDistil, TrainKD, TrainTestLoops
from utils.arguments_from_experiment import hyperparameters_classification #, load_yaml, build_record_folder, get_args
from utils.compare_teacher_student import compare_student_teacher, summary_func

device = "cuda" if torch. torch.cuda.is_available() else "cpu"
print(f"\n DEVICE IN USE: {device} \n")
time.sleep(1)

def distillation_classification(args):
	
# Setting hyperparameters
	#batch_size, num_workers, epochs, learning_rate, loss_func, optimizer_func, temperature, soft_target_loss_weight, ce_loss_weight = hyperparameters_classification()

	# Loading data
	train_loader, test_loader, len_class_names, class_to_idx = Load_Standard_Datasets.Load_CIFAR_10(args.batch_size, args.num_workers)

	# Defining the teachers and the students
	teacher = select_teacher(model=args.teacher,num_classes=len_class_names).to(device)  # To insert the function to select the network --> Change it for a loop that can run several configurations

	student = LightNN(num_classes=len_class_names, teacher_hidden_size=teacher.hidden_size).to(device)

	# BULD IN THIS PART THE STUDENT VARIABLES ACCORDING THE DISTILLATION STEP THAT IT IS PERFORMING.

	new_student = LightNN(num_classes=len_class_names, teacher_hidden_size=teacher.hidden_size).to(device)  # this variable is used to compare the performances before and after distilate

	#print(f"\nThe Teacher Network: {teacher.classifier}\n")   # Check how to can read the Network Nane --> Check in the pre-trained models before # Check how to 
	# use this .classifier in other networks
	print("The Teacher Network")
	summary_func(model=teacher)
	time.sleep(0.5)
	print(f"\nThe Student Network: {student}\n")
	#print("\nThe Student Network\n")
	#summary_func(model=student)
	time.sleep(0.5)

	# comparing the parameters of the teacher and student models
	compare_student_teacher(teacher, student)

	# Now call the train function
	print("\n[IN_PROGRESS: ]Now starting the training fo Teacher and Student\n")
	# In the configuiration below the code saves all the loss, and accuracy for training and test in every epoch, and in the end can be used to compare the results
	train_teacher = TrainTestLoops.train_loop(teacher=teacher,
										   student=None,
										   train_loader=train_loader,
										   test_loader=test_loader,
										   epochs=args.epochs,
										   learning_rate=args.learning_rate,
										   loss_func_train=args.loss_func_train,
										   loss_func_out_layer=None,
										   loss_func_soft_label=None,
										   loss_func_hidden_layers=None,
										   optimizer_func_train=args.optimizer_func_train,
										   optimizer_func_out_layer=None,
										   optimizer_func_soft_label=None,
										   optimizer_func_hidden_layers=None,
										   KD=None, # this is only used in distillation step
										   T=None,
										   soft_target_loss_weight=None,
										   ce_loss_weight_out_layer=None,
										   ce_loss_weight_soft_label=None,
										   hidden_rep_loss_weight=None,
										   ce_loss_weight_hidden_layers=None,
										   feature_map_weight=None,
										   step=1, # 1 means train before distil
										   device=device)

	train_student = TrainTestLoops.train_loop(teacher=None,
												  student=student,
												  train_loader=train_loader,
												  test_loader=test_loader,
												  epochs=args.epochs,
												  learning_rate=args.learning_rate,
												  loss_func_train=args.loss_func_train,
												  loss_func_out_layer=None,
												  loss_func_soft_label=None,
												  loss_func_hidden_layers=None,
												  optimizer_func_train=args.optimizer_func_train,
												  optimizer_func_out_layer=None,
												  optimizer_func_soft_label=None,
												  optimizer_func_hidden_layers=None,
												  KD=None,
												  T=None,
												  soft_target_loss_weight=None,
												  ce_loss_weight_out_layer=None,
												  ce_loss_weight_soft_label=None,
												  hidden_rep_loss_weight=None,
												  ce_loss_weight_hidden_layers=None,
												  feature_map_weight=None,
												  step=1, # 1 means train before distil
												  device=device)


	# Now calling the train function to distillate in the output layer
	print("\n[IN_PROGRESS]NOW STARTING THE DISTILLATION PROCESS \n \n")
	print(f"\n[STEP:] First Step: Distillate from the output layer of Classifier using {args.loss_func_train} Loss Function \n")

	# TO CORRECT THE LINE CODE BELOW - I CALLED THE TrainKD Class direct ==> SHOULD CALL TTrainTestLoops.train_loop --> and before it call the TrainKD --> and training loop 
	# Can adapt the function to underestand in each Distillation step it is

	distilate_output_layer = TrainTestLoops.train_loop(teacher=teacher,
													student=student,
													train_loader=train_loader,
													test_loader=test_loader,
													epochs=args.epochs,
													learning_rate=args.learning_rate,
													loss_func_train=None,
													loss_func_out_layer=args.loss_func_out_layer,
													loss_func_soft_label=None,
													loss_func_hidden_layers=None,
													optimizer_func_train=args.optimizer_func_train,
													optimizer_func_out_layer=args.optimizer_func_out_layer,
													optimizer_func_soft_label=None,
													optimizer_func_hidden_layers=None,
													KD=1,  # KD=1 means that it is working in distillation process in output layer
													T=args.temperature,
													soft_target_loss_weight=args.soft_target_loss_weight,
													ce_loss_weight_out_layer=args.ce_loss_weight_out_layer,
													ce_loss_weight_soft_label=None,
													hidden_rep_loss_weight=None,
													ce_loss_weight_hidden_layers=None,
													feature_map_weight=None,
													step=2,
													device=device)	
	
	
	print(f"\n[STEP: ]Second Step: Distillate from Softlabels using {args.loss_func_soft_label} Loss Function")
	distilate_softlabel = TrainTestLoops.train_loop(teacher=teacher,
													student=student,
													train_loader=train_loader,
													test_loader=test_loader,
													epochs=args.epochs,
													learning_rate=args.learning_rate,
													loss_func_train=None,
													loss_func_out_layer=args.loss_func_out_layer,  # IN further versions will be needed to change it to a option that is possible to choise several loss_fn according needs
													loss_func_soft_label=args.loss_func_soft_label,
													loss_func_hidden_layers=None,
													optimizer_func_train=None,
													optimizer_func_out_layer=None,
													optimizer_func_soft_label=args.optimizer_func_soft_label,
													optimizer_func_hidden_layers = None,
													KD=2,  # KD=2 means that it is working in distillation process in output layer
													T=args.temperature,
													soft_target_loss_weight=None,
													ce_loss_weight_out_layer=None,
													ce_loss_weight_soft_label=args.ce_loss_weight_soft_label,
													hidden_rep_loss_weight=args.hidden_rep_loss_weight,
													ce_loss_weight_hidden_layers=None,
													feature_map_weight=None,
													step=2,
													device=device)
	
	print(f"\n [STEP: ] Third Step: Distilate using feature maps from output conv layers. Using {args.loss_func_hidden_layers}")
	distillate_hidden_layers = TrainTestLoops.train_loop(teacher=teacher,
													student=student,
													train_loader=train_loader,
													test_loader=test_loader,
													epochs=args.epochs,
													learning_rate=args.learning_rate,
													loss_func_train=None,
													loss_func_out_layer=args.loss_func_out_layer,  # IN further versions will be needed to change it to a option that is possible to choise several loss_fn according needs
													loss_func_soft_label=None,
													loss_func_hidden_layers=args.loss_func_hidden_layers,
													optimizer_func_train=None,
													optimizer_func_out_layer=None,
													optimizer_func_soft_label=None,
													optimizer_func_hidden_layers = args.optimizer_func_hidden_layers,
													KD=3,  # KD=2 means that it is working in distillation process in output layer
													T=args.temperature,
													soft_target_loss_weight=None,
													ce_loss_weight_out_layer=None,
													ce_loss_weight_soft_label=None,
													hidden_rep_loss_weight=None,
													ce_loss_weight_hidden_layers=args.ce_loss_weight_hidden_layers,
													feature_map_weight=args.feature_map_weight,
													step=2,
													device=device)	
	
def distillation_detection():
	print("HERE WILL BE DEVELOPED THE CODE TO DISTILATE KNOWLEDGE IN DETECTION TASKS")