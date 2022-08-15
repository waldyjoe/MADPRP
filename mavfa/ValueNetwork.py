import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants.Settings import FC1_UNITS, FC2_UNITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



class Encoder(nn.Module):
    """
    Encoder network for the schedule of one sector/agent (multiple sectoral agents)
    """

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

    def __reduce__(self):
        deserializer = Encoder
        serialized_data = (self.seed, self.fc1, self.fc2, self.fc3,)
        return deserializer, serialized_data


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


class LocalValueNetwork(nn.Module):
    """
    Local V Network
    """

    def __init__(self, input_dim, output_dim, fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        super(LocalValueNetwork, self).__init__()

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


class GlobalValueNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, n_agents, state_size, output_dim, seed, area_size, subagent_dim, embedding_dim, shift_size,
                 encoding_size, hidden_dim_attn, hidden_dim_out, encoder_fc1_unit, encoder_fc2_unit):
        super(GlobalValueNetwork, self).__init__()

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

        # 3 constitutes 2 additional dummy patrol areas and 1 one-hot vector representation of the schedules
        # self.subagent_dim * self.encoding_size  = number of subagents x the dim of the encoded schedules
        encoded_state_dim = self.state_size - 3 + self.subagent_dim * self.encoding_size
        self.attention_1layer = AttentionModel(self.n_agents, encoded_state_dim, hidden_dim_attn, hidden_dim_out)
        self.attention_2layer = AttentionModel(self.n_agents, hidden_dim_out, hidden_dim_attn, hidden_dim_out)
        self.local_vnetwork = LocalValueNetwork(hidden_dim_out, output_dim)
        input_mix_dim = 1 + self.area_size - 2 + n_agents * output_dim
        self.mix_network = MixingNetwork(input_mix_dim, output_dim, n_agents)

    # def __reduce__(self):
    #     deserializer = GlobalValueNetwork
    #     serialized_data = (self.n_agents , self.state_size, self.output_dim, self.seed, self.area_size,
    #                        self.subagent_dim, self.embedding_dim, self.shift_size, self.encoding_size,
    #                        self.hidden_dim_attn, self.hidden_dim_out, self.encoder_fc1_units, self.encoder_fc2_units,)
    #     return deserializer, serialized_data

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
        # for agent in range(self.n_agents):
        #
        #     if agent == 0:
        #         encoded_joint_states = encoded_local_states[agent]
        #         print(encoded_local_states[agent].shape)
        #     else:
        #         encoded_joint_states = torch.stack((encoded_joint_states, encoded_local_states[agent]), 1)
        encoded_joint_states = torch.stack(encoded_local_states, 1)
        # print("Encoded joints states: ", encoded_joint_states.shape)

        # 2 layers of attention model
        # print(mask)
        mask = torch.tensor(mask).to(device)
        # print(mask)
        # mask = torch.from_numpy(mask).to(device)
        # print(mask)
        # sys.exit()
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

class GlobalValueNetworkNoComms(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, n_agents, state_size, output_dim, seed, area_size, subagent_dim, embedding_dim, shift_size,
                 encoding_size, encoder_fc1_unit, encoder_fc2_unit):
        super(GlobalValueNetworkNoComms, self).__init__()

        self.n_agents = n_agents
        self.state_size = state_size  # input dim of the joint state
        self.output_dim = output_dim  # this should be 1 (i.e. the value function)
        self.seed = torch.manual_seed(seed)
        self.area_size = area_size
        self.subagent_dim = subagent_dim  # maximum number of subagent in a sector
        self.embedding_dim = embedding_dim  # For embedding the one-hot vector of each patrol location
        self.shift_size = shift_size  # Length of shift
        self.encoding_size = encoding_size  # encoding size of a schedule of an agent
        # self.hidden_dim_attn = hidden_dim_attn  # hidden dim for attention module
        # self.hidden_dim_out = hidden_dim_out  # hidden dim for the output of attention module
        self.encoder_fc1_units = encoder_fc1_unit
        self.encoder_fc2_units = encoder_fc2_unit

        self.embedding = nn.Linear(in_features=area_size, out_features=embedding_dim, bias=False)
        self.encoder = Encoder(input_size=shift_size, output_size=encoding_size, seed=seed, fc1_unit=encoder_fc1_unit,
                               fc2_unit=encoder_fc2_unit)

        # 3 constitutes 2 additional dummy patrol areas and 1 one-hot vector representation of the schedules
        # self.subagent_dim * self.encoding_size  = number of subagents x the dim of the encoded schedules
        encoded_state_dim = self.state_size - 3 + self.subagent_dim * self.encoding_size
        # self.attention_1layer = AttentionModel(self.n_agents, encoded_state_dim, hidden_dim_attn, hidden_dim_out)
        # self.attention_2layer = AttentionModel(self.n_agents, hidden_dim_out, hidden_dim_attn, hidden_dim_out)
        self.local_vnetwork = LocalValueNetwork(encoded_state_dim, output_dim)
        input_mix_dim = 1 + self.area_size - 2 + n_agents * output_dim
        self.mix_network = MixingNetwork(input_mix_dim, output_dim, n_agents)

    # def __reduce__(self):
    #     deserializer = GlobalValueNetwork
    #     serialized_data = (self.n_agents , self.state_size, self.output_dim, self.seed, self.area_size,
    #                        self.subagent_dim, self.embedding_dim, self.shift_size, self.encoding_size,
    #                        self.hidden_dim_attn, self.hidden_dim_out, self.encoder_fc1_units, self.encoder_fc2_units,)
    #     return deserializer, serialized_data

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
        # for agent in range(self.n_agents):
        #
        #     if agent == 0:
        #         encoded_joint_states = encoded_local_states[agent]
        #         print(encoded_local_states[agent].shape)
        #     else:
        #         encoded_joint_states = torch.stack((encoded_joint_states, encoded_local_states[agent]), 1)
        encoded_joint_states = torch.stack(encoded_local_states, 1)
        # print("Encoded joints states: ", encoded_joint_states.shape)

        # 2 layers of attention model
        # print(mask)
        # mask = torch.tensor(mask).to(device)
        # print(mask)
        # mask = torch.from_numpy(mask).to(device)
        # print(mask)
        # sys.exit()
        # h1 = self.attention_1layer(encoded_joint_states, mask)
        # h2 = self.attention_2layer(h1, mask)
        local_values = self.local_vnetwork(encoded_joint_states)
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

