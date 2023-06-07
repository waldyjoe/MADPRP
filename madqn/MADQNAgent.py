import sys

import numpy as np
import pickle
import random
from collections import namedtuple, deque

from constants.Settings import T
##Importing the model (function approximator for Q-table)
from dqn.QNetwork import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 20  # how often to update the network

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class MADQNAgent():
    """Interacts with and learns form environment."""

    def __init__(self, pre_trained, state_size, output_dim, seed, area_size, subagents_count, encoding_size,
                 embedding_dim=1, shift_size=len(T), trained_parameters=None, imported_memory=[]):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.pre_trained = pre_trained
        self.state_size = state_size  # input dim of the joint state
        self.output_dim = output_dim  # this should be 1 (i.e. the value function)
        self.seed = random.seed(seed)
        self.area_size = area_size
        self.subagents_count = subagents_count  # maximum number of subagent in a sector
        self.encoding_size = encoding_size  # encoding size of a schedule of an agent
        self.embedding_dim = embedding_dim
        self.shift_size = shift_size

        self.qnetwork = QNetwork(state_size, output_dim, seed, area_size, subagents_count, embedding_dim, shift_size,
                                 encoding_size, encoder_fc1_unit=128, encoder_fc2_unit=64).to(device)
        self.qnetwork_target = QNetwork(state_size, output_dim, seed, area_size, subagents_count, embedding_dim,
                                        shift_size, encoding_size, encoder_fc1_unit=128, encoder_fc2_unit=64).to(device)

        # if not a new run, initialise the weights
        if trained_parameters:
            self.qnetwork.load_state_dict(trained_parameters)
            self.qnetwork_target.load_state_dict(trained_parameters)

        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(output_dim, BUFFER_SIZE, BATCH_SIZE, seed)

        if imported_memory:
            self.memory.import_memory(imported_memory)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                loss = self.learn(experience, GAMMA)
                return loss

        return None

    def act(self, state, eps=0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = np.array([state])
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Epsilon -greedy action selction
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.output_dim))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        # shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork, self.qnetwork_target, TAU)

        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def get_network(self):
        return self.qnetwork

    def get_memory(self):
        return self.memory

    def get_t_step(self):
        return self.t_step


class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "done"])
        self.seed = random.seed(seed)

        # For exporting
        self.memory_list = deque(maxlen=10 * batch_size)


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)
        self.memory_list.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
        #     device)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def import_memory(self, imported_memory):

        for e in imported_memory:
            self.add(e[0], e[1], e[2], e[3], e[4])

    def get_memory_list(self):
        return list(self.memory_list)