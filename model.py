import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model(filename):
    """Loads a model from file and returns a QNetwork object."""
    checkpoint = torch.load(filename)
    model = QNetwork(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, fc_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list): a list of nodes per hidden layer.
        """
        super(QNetwork, self).__init__()
        units_list = [state_size] + fc_units + [action_size]
        pairs = zip(units_list[:-1], units_list[1:])
        self.linear_layers = [nn.Linear(pair[0], pair[1]) for pair in pairs]
        for idx, layer in enumerate(self.linear_layers[:-1]):
            self.add_module("hidden_layer" + str(idx), layer)
        self.add_module("output_layer", self.linear_layers[-1])

    def forward(self, state):
        """Build a network that maps state -> action values.
        Params
        ======
            state (array): current environment state.

        Returns
        =======
            array: current state-action-values.
        """
        x = state
        for layer in self.linear_layers[:-1]:
            x = F.relu(layer(x))
        x = self.linear_layers[-1](x)
        return x

    def save_model(self, filename='model.pt'):
        """Saves a model to file (architecture + state_dict (weights)).

        Params
        ======
            filename (str): model filename (defaults to "model.pt")
        """
        checkpoint = {
            'input_size': self.linear_layers[0].in_features,
            'output_size': self.linear_layers[-1].out_features,
            'hidden_layers': [layer.out_features for layer in self.linear_layers[:-1]],
            'state_dict': self.state_dict()}
        torch.save(checkpoint, filename)

    def save_checkpoint(self, filename='checkpoint.pth'):
        """Saves the model weights to file only.

        Params
        ======
            filename (str): checkpoint filename (defaults to "checkpoint.pt")
        """
        torch.save(self.state_dict(), filename)
