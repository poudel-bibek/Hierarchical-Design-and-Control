import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
import math

######## MLP model ########
class MLPActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device, **kwargs):
        """
        A simple MLP Actor-Critic network 
        Since negative values can sparingly occur, use leaky ReLU.
        # MLP network Param count: ~around 138,000

        Since I expect the output to be binary, I need to apply the sigmoid somewhere. 
        The network needs to understand that the 10 choices are binary. (Done in the act function)

        The input is flat
        TODO: Regularization: Apply Dropout and Batch Normalization after the shared layers.
        """
        super(MLPActorCritic, self).__init__()
        self.device = device
        
        # hidden_dim = kwargs.get('hidden_dim', 256) #If not specified, default to 256
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        ).to(device)
        
        # Actor-specific layers
        self.actor_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_dim), # Decided not to use sigmoid here but instead later in the act function.
        ).to(device)
        
        # Critic-specific layers
        self.critic_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
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

        state_tensor = torch.FloatTensor(state.flatten()).to(self.device) # Because MLP receives a flattened input.
        action_logits = self.actor(state_tensor)  # outputs logits for each binary decision
        action_probs = torch.sigmoid(action_logits) # convert these logits to probabilities
        dist = Bernoulli(action_probs) # create a Bernoulli distribution using these probabilities
        action = dist.sample() # sample from this distribution to get our binary actions.
        return action.long(), dist.log_prob(action).sum(-1) #  return the actions and the sum of their log probabilities (sum along the last dimension)
        # TODO: check the validity of the sum operation. 

    def evaluate(self, states, actions):
        action_logits = self.actor(states)
        action_probs = torch.sigmoid(action_logits)
        dist = Bernoulli(action_probs)
        
        action_logprobs = dist.log_prob(actions.float())
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        
        return action_logprobs.sum(-1), state_values, dist_entropy

    def param_count(self, ):
        """
        Return a dict
        """
        actor_params = sum(p.numel() for p in self.actor_layers.parameters())
        critic_params = sum(p.numel() for p in self.critic_layers.parameters())
        shared_params = sum(p.numel() for p in self.shared_layers.parameters())
        
        return {
            "actor_total": actor_params + shared_params,
            "critic_total": critic_params + shared_params,
            "total": actor_params + critic_params + shared_params,
            "shared": shared_params
        }

######## CNN model ########
class CNNActorCritic(nn.Module):
    def __init__(self, in_channels, action_dim, device, **kwargs):
        """
        CNN Actor-Critic network with configurable size (designed to be compatible with hyper-parameter tuning)
        we are applying conv2d, the state should be 2d with a bunch of channels.
        Choices: 
            Config: Small: 
                4 Conv layers
                3 Linear layers

            Config: Medium:
                6 Conv layers
                4 Linear layers

        Regularization: Dropout and Batch Norm (mitigation of internal covariate shift)
        Conservatively using pooling layers. Every piece of information is important, however we also want to avoid overfitting and keep parameters modest. 
        Dilation: For the first layer, experiment with dilation. (Disabled for now)

        During hyper-param sweep, the model size changes based on one of the dimension of the input (action_duration). 
        Even at high action durations, the model size is around 4.1M parameters. 
        """
        super(CNNActorCritic, self).__init__()
        self.device = torch.device(device)
        self.in_channels = in_channels
        
        self.action_duration = kwargs.get('action_duration')
        self.per_timestep_state_dim = kwargs.get('per_timestep_state_dim')
        model_size = kwargs.get('model_size', 'medium')
        kernel_size = kwargs.get('kernel_size', 3)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        padding = kernel_size // 2

        if model_size == 'small':
            self.shared_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Dropout(dropout_rate)
            ).to(self.device)
            hidden_dim = 128

        else:  # medium
            self.shared_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Added pooling layer
                nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Dropout(dropout_rate)
            ).to(self.device)
            hidden_dim = 256

        # Calculate the size of the flattened CNN output
        with torch.no_grad():
            sample_input = torch.zeros(1, in_channels, self.action_duration, self.per_timestep_state_dim).to(self.device) # E.g., (1,1,10,74) batch size of 1, 1 channel, 10 timesteps, 74 state dims
            cnn_output_size = self.shared_cnn(sample_input).shape[1]
            #print(f"\n\nCNN output size: {cnn_output_size}\n\n")

        # Actor-specific layers
        self.actor_layers = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, action_dim)
        ).to(device)
        
        # Critic-specific layers
        self.critic_layers = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        ).to(device)

    def actor(self, state,):
        shared_features = self.shared_cnn(state)
        return self.actor_layers(shared_features)
    
    def critic(self, state):
        shared_features = self.shared_cnn(state)
        return self.critic_layers(shared_features)
    
    def act(self, state):
        """
        Select an action based on the current state
        """
        state_tensor = state.to(self.device).reshape(1, self.in_channels, self.action_duration, self.per_timestep_state_dim)
        action_logits = self.actor(state_tensor)
        action_probs = torch.sigmoid(action_logits)
        dist = Bernoulli(action_probs)
        action = dist.sample().squeeze(0)
        #print(f"\n\nAction: {action}\n\n")
        return action.long(), dist.log_prob(action).sum(-1)

    def evaluate(self, states, actions):
        action_logits = self.actor(states)
        action_probs = torch.sigmoid(action_logits)
        dist = Bernoulli(action_probs)
        
        action_logprobs = dist.log_prob(actions.float())
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        
        return action_logprobs.sum(-1), state_values, dist_entropy

    def param_count(self, ):
        """
        Return a dict
        """
        actor_params = sum(p.numel() for p in self.actor_layers.parameters())
        critic_params = sum(p.numel() for p in self.critic_layers.parameters())
        shared_params = sum(p.numel() for p in self.shared_cnn.parameters())
        
        return {
            "actor_total": actor_params + shared_params,
            "critic_total": critic_params + shared_params,
            "total": actor_params + critic_params + shared_params,
            "shared": shared_params
        }

######## Transformer model ########
