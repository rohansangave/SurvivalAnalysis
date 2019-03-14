from torch import nn 

class DeepSurv(nn.Module):
    def __init__(self, n_in, hidden_layers_sizes = None, activation = "rectify",
        dropout = None,
        batch_norm = False,
        standardize = False,
        momentum = 0.1
        ):
        
        super(DeepSurv, self).__init__()

        self.layers = []

        # Default Standardization Values: mean = 0, std = 1
        #nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)

        hidden_layers_sizes = [n_in] + hidden_layers_sizes + [1]


        if activation == 'rectify':
            activation_fn = nn.ReLU()
        elif activation == 'selu':
            activation_fn = nn.SeLU()
        else:
            raise IllegalArgumentException("Unknown activation function: %s" % activation)

        # Construct Neural Network
        for i in range(len(hidden_layers_sizes)-2):
            self.layers.append(nn.Linear(hidden_layers_sizes[i],hidden_layers_sizes[i+1]))
            self.layers.append(activation_fn)

            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_layers_sizes[i+1], eps=1e-05, momentum=momentum, affine=True, track_running_stats=True))

            if dropout:
                self.layers.append(nn.Dropout(dropout, inplace=True))

        self.layers.append(nn.Linear(hidden_layers_sizes[-2], hidden_layers_sizes[-1]))
        self.network = nn.Sequential(*(self.layers))
        
    def forward(self, x):
#         x = x.view(-1, self.size_list[0]) # Flatten the input
        return self.network(x)