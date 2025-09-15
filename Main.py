# A code to perform Knowledge Distilation - For classfification tasks

from execute_tasks import distillation_classification, distillation_detection
from utils.custom_logger import timeLogger
from utils.arguments_from_experiment import get_args, load_yaml, build_record_folder


# Unkoment the line above when running in A GPU training - to deploy it in a hardare accelerator
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" 

def main(args, tlogger):
	tlogger.print("DEFINING THE TASK TO BE PERFORMED \n")
	tlogger.print("1- Distillate for classification \n 2- Distilate for Detection\n")
	tlogger.print("Type 3 to exit \n")
	task = int(input("Select the task: \n"))

	while True:

		if task == 1:
			distilate = distillation_classification(args)  # run distilation for classification

		elif task == 2:
			distilate = distillation_detection(args)
		
		elif task == 3:
			break

		else:
			print("Wrong option, please type again")
		
		tlogger.print("DEFINING THE TASK TO BE PERFORMED \n")
		tlogger.print("1- Distillate for classification \n 2- Distilate for Detection\n")
		tlogger.print("Type 3 to exit \n")
		task = int(input("Select the task: \n"))

if __name__ == "__main__":
	#tlogler = timeLogger()  # not for now
	tlogger = timeLogger()

	tlogger.print("Reading the Hyperparameters\n")
	args = get_args()
		
	assert args.c !="", "Please Provide hyperparameters.yaml file"
	print(f"Hyperparameters: {args}")
	
	load_yaml(args, args.c)
	build_record_folder(args)

	tlogger.print()

	main(args,tlogger)
