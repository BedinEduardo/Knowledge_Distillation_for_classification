# a function to compare the parameters of student and teacher

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import time

def compare_student_teacher(teacher, sutudent):
	total_parameters_teacher = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
	total_parameters_student = "{:,}".format(sum(p.numel() for p in sutudent.parameters()))
	print(f"Teacher Parameters: {total_parameters_teacher}")
	time.sleep(1)
	print(f"Teacher Parameters: {total_parameters_student}")
	time.sleep(1)
	