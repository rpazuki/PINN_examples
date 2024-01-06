import itertools
import torch
import torch.nn as nn

def XY_gradient(outputs, xy_inputs):
    dxy  = torch.autograd.grad(outputs, xy_inputs, torch.ones_like(outputs), create_graph=True)[0]# computes dy/dx    
    dx, dy = dxy[:,0], dxy[:,1]
    return dx, dy

def Laplacian(outputs, xy_inputs):
    dx, dy = XY_gradient(outputs, xy_inputs)#dxy[:,0], dxy[:,1]
    dx2, _ =  XY_gradient(dx, xy_inputs)#dxy2[:,0]
    _, dy2 =  XY_gradient(dy, xy_inputs)#dyx2[:,1]
    return dx2 + dy2

class Net_base(nn.Module):   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self):
        pass
        
class Net_dense(Net_base):   
    def __init__(self, layers, activation = nn.Tanh, **kwargs):
        super().__init__(**kwargs)
        self.layers = layers
        self.num_layers = len(layers)
        self.activation = activation
        self.build()        
        
    def build(self):
        """Create the state of the layers (weights)"""                
        self.input_layer = nn.Sequential(*[
                        nn.Linear(self.layers[0], self.layers[1]),
                        self.activation()])
        
        self.hidden_layers = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(self.layers[i], self.layers[i+1]),
                            self.activation()]) for i in range(1, self.num_layers - 2)])
        
        self.output_layer = nn.Linear(self.layers[-2], self.layers[-1])
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class Net_dense_normalised(Net_dense):
    def __init__(self, layers, lb, ub, activation = nn.Tanh, **kwargs):
        self.lb = lb
        self.ub = ub
        super().__init__(layers, activation, **kwargs)

    def forward(self, x):
        # Map the inputs to the range [-1, 1]
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return torch.log(1.0 + torch.exp(x))