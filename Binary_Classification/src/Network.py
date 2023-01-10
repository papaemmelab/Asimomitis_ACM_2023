from torch import nn
import torch.nn.functional as F
import torch


class Network(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_layers_size, output_size,activation,dropout_layers,dropout_rate,device):

        super(Network,self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layers_size[0])])
        self.layers.extend([nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]) for i in range(1, num_hidden_layers)])
        self.layers.append(nn.Linear(hidden_layers_size[num_hidden_layers-1], output_size))

        if dropout_rate!=0.0:
            for pos in range(0,len(dropout_layers)):

                self.layers.insert(dropout_layers[pos]+pos+1,nn.Dropout(p=dropout_rate))

        self.activation = activation
        self.device = device

    def forward(self, x):

        for i in range(0,len(self.layers)-1):

            if not isinstance(self.layers[i], nn.Dropout):

                if self.activation=='relu':
                    x = F.relu(self.layers[i](x))
                elif self.activation=='sigmoid':
                    x = torch.sigmoid(self.layers[i](x))
                elif self.activation=='tanh':
                    x = torch.tanh(self.layers[i](x))

            else:

                x = self.layers[i](x)

        x = torch.sigmoid(self.layers[len(self.layers)-1](x))

        return x