# A code to load the hyperparameters used in the codes to distillate knowledge
import time
import yaml
import os
import argparse

def hyperparameters_classification():
	print("INSERT THE HYPER-PARAMETERS \n")
	batch_size = int(input("Insert the batch size value (int): \n"))

	num_workers = int(input("Insert the num of workers (int): \n"))

	epochs = int(input("Insert the Number of Epcohs to train (int): \n"))

	learning_rate = float(input("Insert the Learning Rate (float): \n"))

	print("DEFINE THE LOSS FUNCTION: \n")
	print("1. CrossEntropyLoss")
	loss_func = int(input("Type the loss function value: \n"))

	print("DEFINE THE Optimizer Function: \n")
	print("1. Adam Optimizer")
	optimizer_func = int(input("Type the loss function value: \n"))

	#temperature = float(input("Type the Temperature to control the smoothness of the output distribution: \n"))
	temperature = 2
	temperature = float(temperature)

	#soft_target_loss_weight = float(input("Type the soft target loss weight: \n"))  #(A weight assigned to the extra objective we’re about to include): "))
	soft_target_loss_weight = 0.6
	soft_target_loss_weight = float(soft_target_loss_weight)
		
	#ce_loss_weight = 1 - soft_target_loss_weight
	#ce_loss_weight = float(input("Type the ce_loss_weight: \n"))
	ce_loss_weight = 0.4
	ce_loss_weight = float(ce_loss_weight)
	#print(f"The ce_loss_weight is 1 - soft_target_loss_weight: {ce_loss_weight} \n A weight assigned to the extra objective we’re about to include")

	print(f"Type Soft_traget_loss_weight: {type(soft_target_loss_weight), soft_target_loss_weight} \n")
	print(f"Type Ce_loss_weight: {type(ce_loss_weight), ce_loss_weight} \n")
	time.sleep(1)

	return batch_size, num_workers, epochs, learning_rate, loss_func, optimizer_func, temperature, soft_target_loss_weight, ce_loss_weight

# A code to read the yaml file that contains the hyperparameters.
def load_yaml(args, yml):  # Load the arguments of the yaml file
	with open(yml, 'r', encoding='utf-8') as fyml:
		dic = yaml.load(fyml.read(), Loader=yaml.Loader)
		for k in dic:
			setattr(args, k, dic[k])  # setattr() is a function in Python used to build-in utility to set the value of a name attributed of na object dynamically..

def build_record_folder(args):  # to save the results in a specific folder - to use this code in furthers updates
	if not os.path.isdir("./records/"):
		os.mkdir("./records")
	
	args.save_dir = "./records" + args.project_name + "/" + args.exp_name
	os.makedirs(args.save_dir, exist_ok=True)
	os.makedirs(args.save_dir + "backup/", exist_ok=True)

def get_args(with_deepseed: bool=False): # get the arguments in .yaml file
	parser = argparse.ArgumentParser("Knowledge Distilation for Classification Tasks - for now")

	parser.add_argument("--c", default="hyperparameters.yaml", type=str, help="config file path") 
	parser.add_argument("--project_name", default="")
	parser.add_argument("--exp_name", default="")
	parser.add_argument("--teacher", default="DeepNN", type=str)
	parser.add_argument("--student", default="lightNN", type=str)
	parser.add_argument("--batch_size", default=2, type=int)
	parser.add_argument("--num_workers", default=1, type=int)
	parser.add_argument("--epochs", default=2, type=int)
	parser.add_argument("--learning_rate", default=0.001, type=float)
	parser.add_argument("--loss_func_train", default=1, type=int)  # this is a variable that will be choosen in the code - further adapt it
	parser.add_argument("--loss_func_out_layer", default=1, type=int)
	parser.add_argument("--loss_function_soft_label", default=1, type=int)
	parser.add_argument("--optimizer_func_train", default=1, type=int)
	parser.add_argument("--optimizer_func_out_layer", default=1, type=int)
	parser.add_argument("--optimizer_func_soft_label", default=1, type=int)
	parser.add_argument("--optimizer_func_hidden_layers", default=1, type=int)
	parser.add_argument("--temperature", default=1, type=float)
	parser.add_argument("--soft_target_loss_weight", default=0.5, type=float)
	parser.add_argument("--ce_loss_weight_out_layer", default=0.5 ,type=float)
	parser.add_argument("--ce_loss_weight_soft_label", default=0.5, type=float)
	parser.add_argument("--hidden_rep_loss_weight", default=0.5, type=float)
	parser.add_argument("--ce_loss_weight_hidden_layers", default=0.5, type=float)
	parser.add_argument("--feature_map_weight", default=0.5, type=float)

	
	if with_deepseed:
		import deepseed
		parser = deepseed.add_config_arguments(parser)
	
	args = parser.parse_args()
	

	return args

