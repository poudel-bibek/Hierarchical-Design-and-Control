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
        The key idea is that actor outputs logits for each binary decision, then we convert these logits to probabilities and sample from this distribution to get our binary actions.
        The choice of using Bernoulli distribution is because we want to model the probability of a binary event (e.g., turn ON and OFF).
        We need to return the sum of log probabilities to be used in the PPO loss function. 

        # TODO: check the validity of the sum operation. (Done)
        The validity of the sum operation depends on the intended interpretation of the action space. Let's consider two scenarios:

        a) Independent Binary Actions:
        If each of the (e.g., 10) binary choices is truly independent (e.g., selecting multiple items from a list), then summing the log probabilities is valid. This sum represents the log probability of the entire action vector.
        In this scenario, we can control traffic light for NS and EW independently. This means we can have all combinations: both red, both green, or one red and one green.
        Actions:
            Action 1: NS light (0 = red, 1 = green)
            Action 2: EW light (0 = red, 1 = green)

            Example:
            Let's say our model predicts:

            P(NS = green) = 0.7
            P(NS = red) = 0.3
            P(EW = green) = 0.4
            P(EW = red) = 0.6

            If we sample and get [1, 0] (NS green, EW red):
            Log probability = log(0.7) + log(0.6) = -0.71
            This sum of log probabilities is valid because the decisions for NS and EW are independent. The total log probability represents how likely the model was to choose this specific combination.

        b) Mutually Exclusive Actions:
        If the (e.g., 10) binary choices are meant to be mutually exclusive (e.g., selecting one out of 10 options), then summing the log probabilities isn't the correct approach. In this case, we'd typically use a Categorical distribution instead of multiple Bernoulli distributions.
        Instead that we could only choose one direction to be green at a time, and the other must be red.
        Actions:

            0: NS green, EW red
            1: NS red, EW green

            In this case, we'd use a Categorical distribution:
            P(NS green, EW red) = 0.6
            P(NS red, EW green) = 0.4

            If we choose NS green, EW red:
            Log probability = log(0.6) = -0.51
            Here, summing log probabilities wouldn't make sense because we're making a single choice between mutually exclusive options.

        In our actual traffic light scenario:

        We have multiple independent binary decisions (one for each traffic light and crosswalk).
        Each decision (turn a light ON or OFF) doesn't affect the others directly.
        We can have any combination of lights ON or OFF.

        The actions for different traffic lights and crosswalks are separate decisions, so treating them as independent. 
        Given that the actions are independent binary decisions, summing the log probabilities is a valid operation.
        """

        state_tensor = torch.FloatTensor(state.flatten()).to(self.device) # Because MLP receives a flattened input.
        action_logits = self.actor(state_tensor)  # outputs logits for each binary decision
        action_probs = torch.sigmoid(action_logits) # convert these logits to probabilities

        # Option a)
        # dist = Bernoulli(action_probs) # create a Bernoulli distribution using these probabilities
        # action = dist.sample() # sample from this distribution to get our binary actions.
        # return action.long(), dist.log_prob(action).sum(-1) #  return the actions and the sum of their log probabilities (sum along the last dimension)
        
        # Option b) 
        dist = Categorical(action_probs) # create a Categorical distribution using these probabilities
        action = dist.sample() # sample from this distribution to get our binary actions.
        return action.long(), dist.log_prob(action) #  return the actions and the sum of their log probabilities (no sum)


    def evaluate(self, states, actions):
        """
        Evaluates a batch of states and actions.
        States are passed to actor to get action logits, using which we get the probabilities and then the distribution. similar to act function.
        Then using the sampled actions, we get the log probabilities and the entropy. 
        Finally, we pass the states to critic to get the state values. (used to compute the value function component of the PPO loss)
        The entropy is used as a regularization term to encourage exploration.
        """
        action_logits = self.actor(states)
        action_probs = torch.sigmoid(action_logits)

        # Option a)
        # dist = Bernoulli(action_probs)
        # action_logprobs = dist.log_prob(actions.float())
        # dist_entropy = dist.entropy()
        # state_values = self.critic(states)
        # return action_logprobs.sum(-1), state_values, dist_entropy

        # Option b)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        return action_logprobs, state_values, dist_entropy
    

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

        # Option a)
        # dist = Bernoulli(action_probs)
        # action = dist.sample().squeeze(0)
        # #print(f"\n\nAction: {action}\n\n")
        # return action.long(), dist.log_prob(action).sum(-1)

        # Option b)
        dist = Categorical(action_probs)
        action = dist.sample().squeeze(0)
        return action.long(), dist.log_prob(action)

    def evaluate(self, states, actions):
        """
        
        """
        action_logits = self.actor(states)
        action_probs = torch.sigmoid(action_logits)

        # Option a)
        # dist = Bernoulli(action_probs)
        # action_logprobs = dist.log_prob(actions.float())
        # dist_entropy = dist.entropy()
        # state_values = self.critic(states)
        # return action_logprobs.sum(-1), state_values, dist_entropy

        # Option b)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        return action_logprobs, state_values, dist_entropy  

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

######## GATv2 model ########
# Used excluisively for the design agent. 

from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

class GATv2ActorCritic(nn.Module):
    """
    GATv2 with edge features.
    """

    def __init__(self, in_channels, hidden_channels, heads, edge_dim, device, action_dim, actions_per_node=2, dropout_rate=0.2, min_thickness=0.1, max_thickness=10.0):
        """
        in_channels: Number of input features per node (e.g., x and y coordinates)
        hidden_channels: Number of hidden features.
        heads: Number of attention heads.
        edge_dim: Number of features per edge
        min_thickness: Minimum thickness of a corsswalk.
        max_thickness: Maximum thickness of a corsswalk.
        action_dim is the max number of proposals. 
        actions_per_node: number of things to propose per node. Each proposal has 2 features: [location, thickness]

        TODO: Why would you apply dropout to the input features?
        # Does it make sense to average node embeddins across the nodes?
        # At every timestep, the actions is a whole bunch of things of max size. Critic has to evaluate all that.
        """

        super(GATv2ActorCritic, self).__init__()

        # First Graph Attention Layer
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim, heads=heads, dropout=dropout_rate)

        # Second Graph Attention Layer
        # Why is concat=False?  When True, the outputs from different attention heads are concatenated resulting in an output of size out_channels * heads.
        # When concat=False, the outputs from different heads are averaged, resulting in an output of size out_channels. This reduces the dimensionality of the output

        # Why heads=1? 
        # Often, multi-head attention is used in earlier layers to capture different aspects of the graph, but the final layer consolidates this information.
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, edge_dim=edge_dim, heads=1, concat=False, dropout=dropout_rate)

        # Linear layer for predicting the number of proposals
        self.num_proposals_layer = torch.nn.Linear(hidden_channels, action_dim)

        # Linear layer for predicting location parameters (mean and log std)
        self.location_layer = torch.nn.Linear(hidden_channels, 2)
        
        # Linear layer for predicting thickness parameters (mean and log std)
        self.thickness_layer = torch.nn.Linear(hidden_channels, 2)

        # Store initialization parameters as instance variables
        self.max_proposals = action_dim
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

        self.activation = F.elu
        self.dropout_rate = dropout_rate

        # Add a critic layer
        self.critic_layer = nn.Sequential(
            # node embedding output is shaped (num_nodes, hidden_channels)
            # mean of the node embeddings is shaped (hidden_channels,)
            # Graph embedding + action dim * 2 (location and thickness for each proposal) to 64
            nn.Linear(hidden_channels + action_dim*actions_per_node, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: (num_nodes, in_channels)
        edge_index: Edge indices (2, num_edges). Denotes the connections between nodes. # e.g., edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
        edge_attr: Edge features (num_edges, edge_dim). 
        """

        x = self.activation(self.conv1(x, edge_index, edge_attr))

        # Apply dropout to hidden representations
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # The same edge_attr is used for both the first and second GAT layers. 
        # Apply second GAT layer with edge features (no activation here as it's the final layer)
        # Why are edge features passed again?
        # Preserving edge information: By passing the edge attributes to each layer, the model can maintain information about the edge features throughout the network depth. 
        # Different learned attention: Each GAT layer learns its own attention mechanism. By providing the edge features to each layer, you allow each layer to learn how to use these features differently
        # Residual-like connections for edges: In a sense, passing edge features to each layer creates a form of residual connection for edge information.         
        x = self.conv2(x, edge_index, edge_attr)

        # The output is going to be shaped (num_nodes, hidden_channels)
        print(f"\nx: {x.shape}\n")

        return x  # Return the final node embeddings. Its still not pased through the linear layers here yet.

    def act(self, x, edge_index, edge_attr):
        """
        Propose upto max_proposals number of crosswalks.
        For use in policy gradient methonds, the log probabilities of the actions are needed.
        """

        # Get node embeddings by passing input through GAT layers
        node_embeddings = self.forward(x, edge_index, edge_attr)

        # Predict the number of proposals
        # dim=0 reduces the 2D tensor (num_nodes, hidden_channels) to a 1D tensor of size (hidden_channels,)
        # hidden_channels is the number of features for each node embedding. By taking the mean along dim=0, we're averaging across all nodes.
        num_proposals_logits = self.num_proposals_layer(node_embeddings.mean(dim=0)) # AVERAGED NODE EMBEDDINGS
        num_proposals_probs = F.softmax(num_proposals_logits, dim=0)  # Convert to probabilities
        # Sample the number of proposals (add 1 to ensure at least 1 proposal)
        num_actual_proposals = torch.multinomial(num_proposals_probs, 1).item() + 1

        # location parameters
        location_params = self.location_layer(node_embeddings)
        location_means = torch.sigmoid(location_params[:, 0])  # Ensure means are between 0 and 1
        # Std values are clamped between -20 and 2. 
        # Very large values of log standard deviation can lead to extremely large gradients during backpropagation, which can destabilize training.
        location_log_stds = location_params[:, 1].clamp(-20, 2)  # Clamp log stds for numerical stability
        # exp(-20) ≈ 2.06e-9 (very small, but not zero), exp(2) ≈ 7.39 (reasonably large, but not enormous)
        location_stds = torch.exp(location_log_stds)  # Convert log stds to stds

        # thickness parameters
        thickness_params = self.thickness_layer(node_embeddings)
        # Scale thickness means to be between min_thickness and max_thickness
        thickness_means = torch.sigmoid(thickness_params[:, 0]) * (self.max_thickness - self.min_thickness) + self.min_thickness
        # Std values are clamped between -20 and 2. 
        # Very large values of log standard deviation can lead to extremely large gradients during backpropagation, which can destabilize training.
        thickness_log_stds = thickness_params[:, 1].clamp(-20, 2)  # Clamp log stds for numerical stability
        # exp(-20) ≈ 2.06e-9 (very small, but not zero), exp(2) ≈ 7.39 (reasonably large, but not enormous)
        thickness_stds = torch.exp(thickness_log_stds)  # Convert log stds to stds

        # sample from the distributions using the reparameterization trick
        # Instead of directly predicting fixed values for thicknesses and locations, use a probability distribution to model the uncertainty in these values.
        # The model assumes that thicknesses and locations follow a Gaussian distribution (characterized by a mean and standard deviation)
        # This also allows for uncertainty quantification i.e., the model can express how confident it is about each prediction.
        thicknesses = torch.normal(thickness_means, thickness_stds)
        locations = torch.normal(location_means, location_stds)

        # Create a padded fixed-sized output. 
        output = torch.full((self.max_proposals, 2), -1.0)
        indices = torch.randperm(x.size(0))[:num_actual_proposals]
        output[:num_actual_proposals, 0] = locations[indices]
        output[:num_actual_proposals, 1] = thicknesses[indices]

        return output, num_actual_proposals 

    def evaluate(self, state, actions):
        """
        Evaluate thevalue of the state-action pair.
        actions is the full action
        """
        x, edge_index, edge_attr = state

        # Get node embeddings
        node_embeddings = self.forward(x, edge_index, edge_attr)
        print(f"\n\nnode_embeddings: {node_embeddings.shape}\n\n")

        # Aggregate node embeddings to get a graph-level embedding
        node_embeddings_mean = node_embeddings.mean(dim=0)
        print(f"\n\nnode_embeddings_mean: {node_embeddings_mean.shape}\n\n")
        
        # Concatenate graph embedding with the actions
        state_action = torch.cat([node_embeddings_mean, actions.flatten()])
        print(f"\n\nstate_action: {state_action.shape}\n\n")

        # Pass through critic layers to get the value
        value = self.critic_layer(state_action)
        print(f"\n\nvalue: {value.shape}\n\n")
        
        return value

    def param_count(self):
        """
        Count the total number of parameters in the model.
        """
        # Shared parameters (GATv2Conv layers)
        shared_params = sum(p.numel() for p in self.conv1.parameters()) + \
                        sum(p.numel() for p in self.conv2.parameters())

        # Actor-specific parameters
        actor_params = sum(p.numel() for p in self.num_proposals_layer.parameters()) + \
                    sum(p.numel() for p in self.location_layer.parameters()) + \
                    sum(p.numel() for p in self.thickness_layer.parameters())

        # Critic-specific parameters
        critic_params = sum(p.numel() for p in self.critic_layer.parameters())

        # Total parameters
        total_params = shared_params + actor_params + critic_params

        return {
            "shared": shared_params,
            "actor_total": shared_params + actor_params,
            "critic_total": shared_params + critic_params,
            "total": total_params
        }


################ EXAMPLE USAGE #################
num_nodes = 10 
num_edges = 20
in_channels = 2 # Number of input features per node (e.g., x and y coordinates)
edge_dim = 2 # Number of features per edge

# Test the GATv2 model with edge features. 
model = GATv2ActorCritic(in_channels=in_channels, 
                         hidden_channels=64, 
                         heads=4, 
                         edge_dim=edge_dim, 
                         device='cpu', 
                         action_dim=10)

# Dummy data
x = torch.rand((num_nodes, in_channels))  
edge_index = torch.randint(0, num_nodes, (2, num_edges))  
edge_attr = torch.rand((num_edges, edge_dim))  

# Dummy input (x, edge_index, edge_attr).
dummy_input = (x, edge_index, edge_attr)
print("Input data: ")
print(f"Node features: {x}\n")
print(f"Edge indices: {edge_index}\n")
print(f"Edge features: {edge_attr}\n")

# print the number of parameters in the model
print(model.param_count())

# print the model's architecture
print(model)

# Generate crosswalk proposals using the model
proposed_crosswalks, num_actual_proposals = model.act(x, edge_index, edge_attr)

print(f"Proposed crosswalks: {proposed_crosswalks}\n")
for i, (location, thickness) in enumerate(proposed_crosswalks):
    if i < num_actual_proposals:
        print(f"Location: {location:.4f}, Thickness: {thickness:.2f}")
    else:
        print(f"Padded proposal: {location:.4f}, {thickness:.2f}")
print(f"\nNumber of actual proposals: {num_actual_proposals}")
print(f"Total number of proposals (including padding): {len(proposed_crosswalks)}\n")

# Evaluate an action
state = (x, edge_index, edge_attr)
action = proposed_crosswalks  # Assuming the first proposed crosswalk is the one to evaluate
print(f"Action: {action}")

value = model.evaluate(state, action)
print(f"Evaluated value of the action: {value.item()}")


