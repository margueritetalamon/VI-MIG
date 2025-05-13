import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, data_dim, hidden_sizes, output_dim, task='classification'):
        super(SimpleMLP, self).__init__()

        self.data_dim = data_dim
        self.hidden_sizes = hidden_sizes
        self.ouput_dim = output_dim
        self.task = task

        # Define layers
        self.layers = nn.ModuleList()

        # Input to first hidden layer
        self.layers.append(nn.Linear(data_dim, hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_dim))

        # Activation function
        self.relu = nn.ReLU()

        # Output activation based on task
        self.task = task
        if task == 'classification':
            self.output_activation = nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=1)
        # For regression, no activation is needed at the output

        # Calculate and store the total number of parameters
        self.param_dim = self._count_parameters()

    def _count_parameters(self):
        """Count and return the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # Forward pass through all but the last layer with ReLU
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))

        # Output layer
        x = self.layers[-1](x)

        # Apply output activation if classification
        if self.task == 'classification':
            x = self.output_activation(x)

        return x

    def get_weights_as_vector(self):
        """Extract all weights as a single 1D vector"""
        weight_vector = []

        for param in self.parameters():
            weight_vector.append(param.data.view(-1))

        return torch.cat(weight_vector)

    def set_weights_from_vector(self, weight_vector):
        """Set all weights from a single 1D vector"""
        # Keep track of consumed weights
        pointer = 0

        for param in self.parameters():
            # Number of weights in this parameter
            num_param = param.numel()

            # Set the parameter data
            param.data = weight_vector[pointer:pointer+num_param].view(param.data.shape)

            # Move the pointer
            pointer += num_param

    def get_gradients_as_vector(self):
        """Extract all gradients as a single 1D vector"""
        grad_vector = []

        for param in self.parameters():
            # Check if gradients exist
            if param.grad is not None:
                grad_vector.append(param.grad.view(-1))
            else:
                # If no gradient, add zeros
                grad_vector.append(torch.zeros_like(param.data.view(-1)))

        return torch.cat(grad_vector)

    def clear_grads(self):
        self.zero_grad()
        # for param in self.parameters():
        #     if param.grad is not None:
        #         param.grad.zero_()

    def clone(self):
        """Create a deep copy of the network with the same architecture and weights"""
        # Create a new instance with the same architecture
        clone_model = SimpleMLP(
            data_dim=self.data_dim,
            hidden_sizes=self.hidden_sizes,
            output_dim=self.ouput_dim,
            task=self.task
        )
        
        # Copy the weights from this model to the clone
        clone_model.load_state_dict(self.state_dict())
        
        return clone_model

