# a function to compare the parameters of student and teacher

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import time

from torchinfo import summary

def compare_student_teacher(teacher, sutudent):
	total_parameters_teacher = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
	total_parameters_student = "{:,}".format(sum(p.numel() for p in sutudent.parameters()))
	print(f"Teacher Parameters: {total_parameters_teacher}")
	time.sleep(1)
	print(f"Teacher Parameters: {total_parameters_student}")
	time.sleep(1)

def summary_func(model: torch.nn.Module):
	
	summary(model=model,
		 input_size=(1,3,224,224),
		 col_names=["input_size","output_size", "num_params","trainable"],
		 row_settings=["var_names"])