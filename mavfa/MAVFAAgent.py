import numpy as np
import pickle
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from mavfa.ValueNetwork import GlobalValueNetwork, GlobalValueNetworkNoComms
from constants.Settings import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class MAVFAAgent():

    def __init__(self, pre_trained, sector_ids, n_agents, state_size, output_dim, seed, area_size, subagent_dim, encoding_size,
                 hidden_dim_attn, hidden_dim_out, embedding_dim=1, shift_size=len(T), trained_parameters=None,
                 imported_memory=[], comms_net=True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.pre_trained = pre_trained
        self.n_agents = n_agents
        self.state_size = state_size  # input dim of the joint state
        self.output_dim = output_dim  # this should be 1 (i.e. the value function)
        self.seed = random.seed(seed)
        self.area_size = area_size
        self.subagent_dim = subagent_dim  # maximum number of subagent in a sector
        self.encoding_size = encoding_size  # encoding size of a schedule of an agent
        self.hidden_dim_attn = hidden_dim_attn  # hidden dim for attention module
        self.hidden_dim_out = hidden_dim_out  # hidden dim for the output of attention module
        self.embedding_dim = embedding_dim  # For embedding the one-hot vector of each patrol location
        self.shift_size = shift_size  # Length of shift

        # Value Function Network
        if comms_net:
            self.vnetwork = GlobalValueNetwork(n_agents, state_size, output_dim, seed, area_size, subagent_dim,
                                               embedding_dim, shift_size, encoding_size, hidden_dim_attn, hidden_dim_out,
                                               encoder_fc1_unit=128, encoder_fc2_unit=64).to(device)
            self.vnetwork_target = GlobalValueNetwork(n_agents, state_size, output_dim, seed, area_size, subagent_dim,
                                                      embedding_dim, shift_size, encoding_size, hidden_dim_attn,
                                                      hidden_dim_out, encoder_fc1_unit=128, encoder_fc2_unit=64).to(device)
        else:
            self.vnetwork = GlobalValueNetworkNoComms(n_agents, state_size, output_dim, seed, area_size, subagent_dim,
                                               embedding_dim, shift_size, encoding_size,
                                               encoder_fc1_unit=128, encoder_fc2_unit=64).to(device)
            self.vnetwork_target = GlobalValueNetworkNoComms(n_agents, state_size, output_dim, seed, area_size, subagent_dim,
                                                      embedding_dim, shift_size, encoding_size, encoder_fc1_unit=128, encoder_fc2_unit=64).to(
                device)

        # if not a new run, initialise the weights
        # if self.new_run != "True":
        #     self.vnetwork.load_state_dict(torch.load("./mavfa/parameter/vfa_parameters_" + str(sector_ids) + ".pth",
        #                                              map_location=device))
        #     self.vnetwork_target.load_state_dict(torch.load(
        #         "./mavfa/parameter/vfa_parameters_" + str(sector_ids) + ".pth", map_location=device))

        if trained_parameters:
            self.vnetwork.load_state_dict(trained_parameters)
            self.vnetwork_target.load_state_dict(trained_parameters)


        self.optimizer = optim.Adam(self.vnetwork.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        if imported_memory:
            self.memory.import_memory(imported_memory)


        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, global_state, action, reward, next_state, next_global_state, done, mask):
        # Save experience in replay memory
        self.memory.add(state, global_state, action, reward, next_state, next_global_state, done)



        self.t_step = (self.t_step + 1)

        #        if len(self.memory) > BATCH_SIZE:
        #            experiences = self.memory.sample()
        #            loss = self.learn(experiences, GAMMA)
        #            return loss

        # Learn every UPDATE_EVERY time steps.
        #        self.t_step = (self.t_step + 1) % LEARN_EVERY

        if (self.t_step) % LEARN_EVERY == 0:
            #        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences, GAMMA, mask)
                return loss

        return None

    def learn(self, experiences, gamma, mask):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
        """
        states, global_states, actions, rewards, next_states, next_global_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        V_targets_next = self.vnetwork_target(next_states, next_global_states, mask).detach().max(1)[0].unsqueeze(1)
        #        print ('Rewards', rewards)

        # Compute Q targets for current states
        V_targets = rewards + (gamma * V_targets_next * (1 - dones))
        #        print ('Q_targets', Q_targets)
        # Get expected Q values from local model
        V_expected = self.vnetwork(states, global_states, mask).gather(1, actions)
        #        print ('Q_expected', Q_expected)
        #        sys.exit('paused')
        # Compute loss
        loss = F.mse_loss(V_expected, V_targets)
        # print(self.t_step, loss.item())

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.t_step % UPDATE_EVERY == 0:
            self.soft_update(self.vnetwork, self.vnetwork_target, TAU)

        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        # target_model.load_state_dict(local_model.state_dict())

    def return_value(self, state, global_state, mask):

        state = np.array([state])
        global_state = np.array([global_state])
        global_state = torch.from_numpy(global_state).float().to(device)

        #        next_state_time = torch.from_numpy(fromNumbertoTensor(normalize_time(to_realtime(state.currentTime)))).float().unsqueeze(0).to(device)
        #        next_state = (next_state,next_state_time)

        self.vnetwork.eval()
        with torch.no_grad():
            action_value = self.vnetwork(state, global_state, mask)
        action_value = np.max(action_value.cpu().data.numpy())
        #        print (state.print_final_route(initial_env))
        #        print (action_value)
        #        sys.exit('paused')
        self.vnetwork.train()
        return action_value

    def get_network(self):
        return self.vnetwork

    def get_memory(self):
        return self.memory

    def get_t_step(self):
        return self.t_step


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "global_state", "action", "reward",
                                                                "next_state", "next_global_state", "done"])
        self.seed = random.seed(seed)
        # For exporting
        self.memory_list = deque(maxlen=10 * batch_size)

    def add(self, state, global_state, action, reward, next_state, next_global_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, global_state, action, reward, next_state, next_global_state, done)
        self.memory.append(e)
        self.memory_list.append((state, global_state, action, reward, next_state, next_global_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        states = np.stack([e.state for e in experiences if e is not None])
        global_states = torch.from_numpy(np.vstack([e.global_state for e in experiences if e is not None])).float().\
            to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = np.stack([e.next_state for e in experiences if e is not None])
        next_global_states = torch.from_numpy(np.vstack([e.next_global_state for e in experiences if e is not None])).\
            float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return states, global_states, actions, rewards, next_states, next_global_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def import_memory(self, imported_memory):

        for e in imported_memory:
            self.add(e[0], e[1], e[2], e[3], e[4], e[5], e[6])

    def get_memory_list(self):
        return list(self.memory_list)


