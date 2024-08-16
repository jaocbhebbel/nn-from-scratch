import numpy as np

class Neuron:
    def __init__(self, isInput):
        self.isInput = isInput
        
        if isInput:
            self.weights = ""
            self.bias = ""
        else:
            self.weights = []
            self.bias = 0
        
        self.z = 0
        self.a = 0

    def activate(self):
        self.a = max(0, self.z)

    def forward_propagate(self):
        self.weights = [
