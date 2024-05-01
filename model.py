import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__() # super() is used to call methods from the parent class
        self.l1 = nn.Linear(input_size, hidden_size)   # input_size is  is the size of the input features
        self.l2 = nn.Linear(hidden_size, hidden_size)  # hidden_size is the number of neurons in the hidden layers
        self.l3 = nn.Linear(hidden_size, num_classes)  # num_classes is the number of output classes
        self.relu = nn.ReLU()  # Activation Function convert any value -ve to zero helping to make high training
    
    def forward(self, x):
        out = self.l1(x)  # di 3bra en el model eli na 48al bi bt3di 3la kol layers bt3ml feed forward
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
    
    # nn.Linear have default bias  and automatically initialized (Weigth)