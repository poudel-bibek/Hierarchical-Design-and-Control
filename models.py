import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        """
        Define the neural networks for both actor and critic
        What activation to use. And how will it affect the output.

        State_dim = 380 (10x38)
        We have no negative values, so we can use ReLU activations.
        # Param count: ~around 138,000
        """
        super(MLPActorCritic, self).__init__()
        self.device = device
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        ).to(device)
        
        # Actor-specific layers
        self.actor_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        ).to(device)
        
        # Critic-specific layers
        self.critic_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Corrected from 16 to 64
        ).to(device)
    
    # Actor Network
    def actor(self, state):
        shared_features = self.shared_layers(state)
        return self.actor_layers(shared_features)
    
    # Critic Network
    def critic(self, state):
        shared_features = self.shared_layers(state)
        return self.critic_layers(shared_features)
    
    def act(self, state):
        """
        Select an action based on the current state
        """
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, states, actions):
        """
        Evaluate the actions given the states
        """
        action_probs = self.actor(states)
        dist = Categorical(logits=action_probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        
        return action_logprobs, state_values, dist_entropy