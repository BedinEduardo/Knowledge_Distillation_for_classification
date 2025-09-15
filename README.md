# Knowledge Distillation Code for Classification Tasks

This is a code developed to perform KD for classification tasks.

This code is still in the development process, but for a while, it loads the CIFAR10 dataset as the base dataset, uses the Cross-Entropy loss function, and the ADAM optimizer in the first stage of the training process.

The code at this moment uses a hands-on DeepNN model and an EfficientNetB0 as teacher models, and a hands-on LightNN model as a student.

The code performs three different KD tasks. The first one uses a loss function (Cross-entropy loss) in the output layer to distillate knowledge between the teacher network and student network, where the student model adjusts its weights according to the balanced loss using the loss output transferred by the teacher.
The second method uses soft-labels to distill knowledge between teacher and student.
And the third one uses feature maps as a way to distill knowledge between the networks.

## Adjusting hyperparameters
In *hyperparameters.yaml* sets the variables according to the desired experiment setup.

## Running the code
To execute the code:
```python3
python3 Main.py
```
And follow the instructions.
