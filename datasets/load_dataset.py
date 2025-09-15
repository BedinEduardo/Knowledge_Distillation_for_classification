# A code responsible to load the datasets
# Starting loading CIFAR-10 Dataset
# Pytorch

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import time

class Load_Standard_Datasets:
	# Unkoment the line above when running in A GPU training - to deploy it in a hardare accelerator
	#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" 
	# device = "cuda" if torch. torch.cuda.is_available() else "cpu"
	# print(f"Device: {device}")
	
	def Load_CIFAR_10(batch_size: int, num_workers: int):
		# preprocessing data for CIFAR10
		transforms_cifar = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # can add another transformation as necessary
			# see Daniel Bourke examples and test for training and test sets - accordingle the conditions
		])
		print(f"\nTransform CIFAR10 Step: {transforms_cifar}")

		# Now loading the CIFAR10 dataset
		train_dataset = datasets.CIFAR10(root='.data', train=True, download=True, transform=transforms_cifar)
		print("\nLoaded the Train set succefuly ")
		test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)
		print("\nLoaded the Test set succefuly")
		time.sleep(1)

		# now defining the TrainLoader and TestLoader
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
		print("\nDefined the TrainLoader succefully")
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		print("\nDefined the TestLoader succefully")
		time.sleep(1)

		# Now getting the classes names
		class_names = train_dataset.classes
		len_class_names = len(class_names)
		print(f"The CIFAR10 Data set has {len_class_names} and the classes are: {class_names}")
		time.sleep(1.5)
		class_to_idx = train_dataset.class_to_idx

		return train_loader, test_loader, len_class_names, class_to_idx
	
	# Below can be added for more datasets -- Stadandards or own
	# it will be necessary to add a Cross-Validation techinique to build the datasets