import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from constants.Settings import FC1_UNITS, FC2_UNITS

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


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


class AttentionModel(nn.Module):

    def __init__(self, node_count, input_dim, hidden_dim, output_dim):
        super(AttentionModel, self).__init__()
        self.fcv = nn.Linear(input_dim, hidden_dim)
        self.fck = nn.Linear(input_dim, hidden_dim)
        self.fcq = nn.Linear(input_dim, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 9e15 * (1 - mask), dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        out = F.relu(self.fcout(out))

        return out


class LocalQNetwork(nn.Module):
    """
    Local V Network
    """

    def __init__(self, input_dim, output_dim, fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        super(LocalQNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_dim)

    def forward(self, local_hidden_state):
        x = F.relu(self.fc1(local_hidden_state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MixingNetwork(nn.Module):
    """
    Mixing Network
    """

    def __init__(self, input_dim, output_dim, n_agents,fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        super(MixingNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim
        self.n_agents = n_agents

        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_dim)

    def forward(self, local_hidden_state, global_state):

        transformed_local_hidden_state = torch.reshape(local_hidden_state, (local_hidden_state.shape[0], -1,))
        x = torch.cat([transformed_local_hidden_state, global_state], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GlobalQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, n_agents, state_size, output_dim, seed, area_size, subagent_dim, embedding_dim, shift_size,
                 encoding_size, hidden_dim_attn, hidden_dim_out, encoder_fc1_unit, encoder_fc2_unit):
        super(GlobalQNetwork, self).__init__()

        self.n_agents = n_agents
        self.state_size = state_size  # input dim of the joint state
        self.output_dim = output_dim  # this should be 1 (i.e. the value function)
        self.seed = torch.manual_seed(seed)
        self.area_size = area_size
        self.subagent_dim = subagent_dim  # maximum number of subagent in a sector
        self.embedding_dim = embedding_dim  # For embedding the one-hot vector of each patrol location
        self.shift_size = shift_size  # Length of shift
        self.encoding_size = encoding_size  # encoding size of a schedule of an agent
        self.hidden_dim_attn = hidden_dim_attn  # hidden dim for attention module
        self.hidden_dim_out = hidden_dim_out  # hidden dim for the output of attention module
        self.encoder_fc1_units = encoder_fc1_unit
        self.encoder_fc2_units = encoder_fc2_unit

        self.embedding = nn.Linear(in_features=area_size, out_features=embedding_dim, bias=False)
        self.encoder = Encoder(input_size=shift_size, output_size=encoding_size, seed=seed, fc1_unit=encoder_fc1_unit,
                               fc2_unit=encoder_fc2_unit)

    def forward(self, joint_states, global_states, mask=None):  # global_state, mask

        # Embed and encode schedules of each agent into its encoded input local state
        encoded_local_states = []

        for agent in range(self.n_agents):
            local_states = joint_states[:, agent:agent + 1]
            np_temp = np.empty(shape=(local_states.shape[0], local_states.shape[2]), dtype='object')
            for row in range(local_states.shape[0]):
                for column2 in range(local_states.shape[2]):
                    np_temp[row][column2] = local_states[row][0][column2]

            encoded_local_states.append(self.embed_encode_schedule_by_agent(np_temp))

        # Stack the encoded local states into joint states shape [batch_size x n_agents x encoded_state_dim]
        encoded_joint_states = torch.stack(encoded_local_states, 1)

        # 2 layers of attention model
        mask = torch.from_numpy(mask).to(device)
        h1 = self.attention_1layer(encoded_joint_states, mask)
        # h2 = self.attention_2layer(h1, mask)
        local_values = self.local_vnetwork(h1)
        global_values = self.mix_network(local_values, global_states)
        # local vnetwork

        return global_values

    def embed_encode_schedule_by_agent(self, local_states):

        schedules_to_encode = local_states[:, -1:]
        embed_encode = []
        for item in schedules_to_encode:
            embed_encode.append(item[0])

        embed_encode = np.array(embed_encode)

        schedules_to_encode = torch.from_numpy(embed_encode).float().to(device)

        embedded_schedules = self.embedding(schedules_to_encode)
        embedded_schedules = torch.flatten(embedded_schedules, start_dim=2)
        encoded_schedules = self.encoder(embedded_schedules)
        encoded_schedules = torch.flatten(encoded_schedules, start_dim=1)

        local_states = np.vstack(local_states[:, :-1]).astype(np.float)
        local_states = torch.from_numpy(local_states).float().to(device)

        encoded_local_states = torch.cat([local_states, encoded_schedules], 1)

        return encoded_local_states

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, output_size, seed, area_size, agent_count, embedding_dim, shift_size, encoding_size,
                 encoder_fc1_unit, encoder_fc2_unit, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.area_size = area_size
        self.seed = torch.manual_seed(seed)

        self.embedding = nn.Linear(in_features=area_size, out_features=embedding_dim, bias=False)
        self.encoder = Encoder(input_size=shift_size, output_size=encoding_size, seed=seed, fc1_unit=encoder_fc1_unit,
                               fc2_unit=encoder_fc2_unit)

        self.agent_count = agent_count
        # print("Shape", state_size - 2 + agent_count * encoding_size + agent_count + 1)

        self.fc1 = nn.Linear(state_size - 2 + agent_count * encoding_size + agent_count + 1, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)

    def forward(self, state):

        # Encode locations
        locations_to_encode = state[:,-2:-1]
        transformed_encoded_locations = []
        for item in locations_to_encode:
            transformed_encoded_locations.append(item[0])

        transformed_encoded_locations = np.array(transformed_encoded_locations)
        locations_to_encode = torch.from_numpy(transformed_encoded_locations).float().to(device)

        embedded_locations = self.embedding(locations_to_encode)
        embedded_locations = torch.flatten(embedded_locations, start_dim=1)

        # Encode schedules
        schedules_to_encode = state[:,-1:]
        transformed_encoded_schedules = []
        for item in schedules_to_encode:
            transformed_encoded_schedules.append(item[0])

        transformed_encoded_schedules =np.array(transformed_encoded_schedules)

        schedules_to_encode = torch.from_numpy(transformed_encoded_schedules).float().to(device)

        state = np.vstack(state[:, :-2]).astype(np.float)

        state = torch.from_numpy(state).float().to(device)


        # # for each agent, encode the time table
        # to_encode = torch.from_numpy(to_encode).float().to(device)
        embedded_schedules = self.embedding(schedules_to_encode)
        embedded_schedules = torch.flatten(embedded_schedules, start_dim=2)
        hidden_state_schedules = self.encoder(embedded_schedules)

        hidden_state_schedules = torch.flatten(hidden_state_schedules, start_dim=1)

        transformed_state = torch.cat([state, embedded_locations, hidden_state_schedules], 1)

        # print(transformed_state.shape)
        # sys.exit()

        x = F.relu(self.fc1(transformed_state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)