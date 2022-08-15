import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants.Settings import FC1_UNITS, FC2_UNITS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# class ValueNetwork(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, seed, fc1_units=FC1_UNITS,fc2_units=FC2_UNITS):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(ValueNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#
#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

class Encoder(nn.Module):

    def __init__(self, input_size=73, output_size=5, seed=None, fc1_unit=128, fc2_unit=64):
        super(Encoder, self).__init__()  ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ValueNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, output_size, seed, area_size, agent_count, embedding_dim, shift_size, encoding_size,
                 encoder_fc1_unit, encoder_fc2_unit, fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        super(ValueNetwork, self).__init__()
        self.area_size = area_size
        self.seed = torch.manual_seed(seed)

        self.embedding = nn.Linear(in_features=area_size, out_features=embedding_dim, bias=False)
        self.encoder = Encoder(input_size=shift_size, output_size=encoding_size, seed=seed, fc1_unit=encoder_fc1_unit,
                               fc2_unit=encoder_fc2_unit)

        self.agent_count = agent_count

        self.fc1 = nn.Linear(state_size - 1 + agent_count * encoding_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)

    def forward(self, state):

        to_encode = state[:,-1:]
        transformed_encode = []
        for item in to_encode:
            transformed_encode.append(item[0])

        transformed_encode =np.array(transformed_encode)

        to_encode = torch.from_numpy(transformed_encode).float().to(device)

        state = np.vstack(state[:, :-1]).astype(np.float)

        state = torch.from_numpy(state).float().to(device)

        # # for each agent, encode the time table
        # to_encode = torch.from_numpy(to_encode).float().to(device)
        embedded = self.embedding(to_encode)
        embedded = torch.flatten(embedded, start_dim=2)
        hidden_state = self.encoder(embedded)

        hidden_state = torch.flatten(hidden_state, start_dim=1)

        transformed_state = torch.cat([state, hidden_state], 1)

        x = F.relu(self.fc1(transformed_state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
