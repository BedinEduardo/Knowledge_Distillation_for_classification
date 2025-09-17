# A code to test the outputs of the Networks
import torch
import torch.nn as nn

from teachers import DeepNN, EfficientNetB0Teacher
from students import LightNN

sample_teacher_input = torch.rand(2,3,224,224)
sample_student_input = torch.rand(2,3,32,32)

teacher = EfficientNetB0Teacher(num_classes=10, pretrained=True)

student = LightNN(num_classes=10, teacher_hidden_size=teacher.hidden_size)

x_teacher, flattened_conv_output_after_pooling_teacher, conv_feature_maps_teacher = teacher(sample_teacher_input)

x_student, flattened_conv_output_student, regressor_output_student = student(sample_student_input)