# A code to test the outputs of the Networks
import torch
import torch.nn as nn

from teachers import DeepNN, EfficientNetB0Teacher
from students import LightNN
from teachers import select_teacher
#from utils.arguments_from_experiment import get_args

#args = get_args()

sample_teacher_input = torch.rand(1,3,224,224)   # DeepNN use 32,32. EfficientNetB0Teacher use 224,224
sample_student_input = torch.rand(1,3,32,32)

print("\nPRINTING TEACHER")
print(f"\nsample_teacher_input: {sample_teacher_input.shape}\n")

teacher = select_teacher(model="EfficientNetB0Teacher",num_classes=10)
teacher_out = teacher(sample_teacher_input)
#print(f"\nTeacher output types: \n", [type(0) for o in teacher_out])

#select_teacher(model=args.teacher,num_classes=10)

print("\n\nPRINTING STUDENT LIGHTNN")
print(f"\nsample_student_input: {sample_student_input.shape}")
student = LightNN(num_classes=10, teacher=teacher, input_size=(3,32,32)) #.to(device)
student_out = student(sample_student_input)
#print(f"\n\nStudent output types: \n", [type(0) for o in student_out])
#x_teacher, flattened_conv_output_after_pooling_teacher, conv_feature_maps_teacher = teacher(sample_teacher_input)

#x_student, flattened_conv_output_student, regressor_output_student = student(sample_student_input)