import torch
import torch.nn as nn
from .db6 import db6_wavelet


# Definición de la red neuronal
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, dilation, translation):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # Primera capa oculta
        self.fc2 = nn.Linear(50, 50)         # Segunda capa oculta
        self.fc3 = nn.Linear(50, input_size) # Capa de salida (misma longitud que la entrada)
        self.dilation = dilation             # Parámetro de dilatación
        self.translation = translation       # Parámetro de traslación

    def forward(self, x):
        # Primera capa oculta con activación sigmoide
        x = torch.sigmoid(self.fc1(x))
        
        # Segunda capa oculta con activación wavelet DB6
        x = db6_wavelet(x, self.dilation, self.translation)
        
        # Capa de salida con activación lineal
        x = self.fc3(x)
        return x