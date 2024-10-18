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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical

class GATv2ActorCritic(nn.Module):
    """
    GATv2 with edge features.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, initial_heads, second_heads, edge_dim, action_hidden_channels, action_dim, gmm_hidden_dim, num_mixtures = 3, actions_per_node=2, dropout_rate=0.2, min_thickness=0.1, max_thickness=10.0):
        """
        in_channels: Number of input features per node (e.g., x and y coordinates)
        hidden_channels: Number of hidden features.
        out_channels: Number of output features.
        initial_heads: Number of attention heads for the first GAT layer.
        second_heads: Number of attention heads for the second GAT layer.
        edge_dim: Number of features per edge
        min_thickness: Minimum thickness of a crosswalk.
        max_thickness: Maximum thickness of a crosswalk.
        action_dim is the max number of proposals. 
        actions_per_node: number of things to propose per node. Each proposal has 2 features: [location, thickness]

        TODO: 
        # At every timestep, the actions is a whole bunch of things of max size. Critic has to evaluate all that.
        # Thickness and location values that are not in the proposal are set to -1 (which is close to minimum of 0.1)
        # model could potentially interpret these as meaningful values. 

        """

        super(GATv2ActorCritic, self).__init__()

        # First Graph Attention Layer. # conv1 should output [num_nodes, hidden_channels * initial_heads]
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim, heads=initial_heads, concat=True, dropout=dropout_rate)# concat=True by default

        # Second Graph Attention Layer Why ever set concat=False?  
        # When True, the outputs from different attention heads are concatenated resulting in an output of size hidden_channels * initial_heads.
        # When concat=False, the outputs from different heads are averaged, resulting in an output of size hidden_channels. This reduces the dimensionality of the output

        # Why heads=1? Often, multi-head attention is used in earlier layers to capture different aspects of the graph, but the final layer consolidates this information.
        # conv2 should output [num_nodes, out_channels * second_heads] (when concat = True)
        # conv2 should output [num_nodes, out_channels] (when concat = False) This loses too much information.
        self.conv2 = GATv2Conv(hidden_channels * initial_heads, out_channels, edge_dim=edge_dim, heads=second_heads, concat=True, dropout=dropout_rate)

        # These layers are passed through the readout layer. 
        #(without the readout layer, the expected input shape here is num_nodes * out_channels * second_heads and num_nodes can be different for each graph and cannot be pre-determined)
        # Linear layer for predicting the number of proposals
        self.num_proposals_layer = torch.nn.Linear(out_channels * second_heads, action_dim)

        # To predict the GMM parameters that make the distribution of the actions.
        # Stacked linear layers for GMM parameters (instead of separate layers for each)
        # Output: num_mixtures * 5 values
        #   - num_mixtures for mix logits (weights of each Gaussian), determines the weight of this Gaussian in the mixture
        #   - num_mixtures * 2 for means (location and thickness), determines the center of the Gaussian 
        #   - num_mixtures * 2 for covariances (diagonal, for simplicity), determines the spread of the Gaussian
        # Using stacked linear layers for joint prediction of all GMM parameters
        self.gmm_layers = nn.Sequential(
            nn.Linear(out_channels * second_heads, gmm_hidden_dim),
            nn.ReLU(),
            nn.Linear(gmm_hidden_dim, num_mixtures * 5)  # 5 = 1 (mix_logit) + 2 (means) + 2 (covs)
        )

        # Store initialization parameters as instance variables
        self.max_proposals = action_dim
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.num_mixtures = num_mixtures
        self.activation = F.elu
        self.dropout_rate = dropout_rate

        # Encoder for processing actions while preserving their structure
        # The raw actions (of shape (max_proposals, 2)) may not be in the most informative format for the critic to evaluate
        # This encoder allows for a richer representation of actions. Since we have max_proposals with padding, we need to make sure the critic focuses on the relevant part of the total actions.
        self.action_encoder = nn.Sequential(
            nn.Linear(self.max_proposals * actions_per_node, action_hidden_channels), # max_proposals * 2
            nn.ReLU(),
            nn.Linear(action_hidden_channels, action_hidden_channels)
        )

        # This layer gets input the graph embedding and the action embedding. 
        self.critic_layer = nn.Sequential(
            # graph/ node embedding output is shaped (out_channels * second_heads) (1D output of the readout layer)
            # Graph embedding + hidden_channels (output of the action encoder) to 64
            nn.Linear(out_channels * second_heads + action_hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # output a single value
        )

        # Temperature parameter (of the softmax function) for controlling exploration in action selection
        # A lower temperature (0.1) makes the distribution more peaked (more deterministic), while a higher temperature (2.0) makes it more uniform (more random).
        #self.temperature = nn.Parameter(torch.ones(1)) # this is a learnable parameter. No need for this to be a learnable. Other mechanisms to control exploration.
        self.temperature = 1.0 # fixed temperature

    def get_gmm_distribution(self, node_embeddings):
        """
        Get the GMM distribution for the node embeddings.
        """
        gmm_params = self.gmm_layers(node_embeddings)
        mix_logits, means, covs = gmm_params.split([self.num_mixtures, self.num_mixtures * 2, self.num_mixtures * 2], dim=-1)
        
        means = means.view(-1, self.num_mixtures, 2)
        covs = F.softplus(covs).view(-1, self.num_mixtures, 2) # Ensure positive covariance

        mix = Categorical(logits=mix_logits) # Categorical distribution for the mixture probabilities
        covariance_matrices = torch.diag_embed(covs) # Create diagonal covariance matrices
        comp = MultivariateNormal(means, covariance_matrices) # Multivariate normal distributions for each component
        gmm = MixtureSameFamily(mix, comp) # Mixture of Gaussians distribution
        return gmm

    def forward(self, x, edge_index, edge_attr, batch):
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

        # Add a readout layer. Use global_mean_pool to agerage across the nodes for each graph in a batch.
        # Consistent way to handle variable-sized graphs.
        x = global_mean_pool(x, batch)

        return x  # Return the final node embeddings. Its still not pased through the linear layers here yet.
    
    def gmm_entropy(self, gmm):
        """
        The entropy measures the uncertainty or randomness in the action selection process given a state.
        - High Entropy: If the policy is highly uncertain about which action to take (i.e., 
        it assigns similar probabilities to multiple actions), the entropy will be high. 
        This encourages exploration because the policy is not overly confident in selecting a single action.

        - Low Entropy: f the policy is very certain about which action to take (i.e., it assigns a high probability to a specific action and low probabilities to others), 
        the entropy will be low, which indicates more deterministic behavior.

        Approximate the entropy of a Gaussian Mixture Model (GMM).
        
        This method uses a Monte Carlo approximation:
        1. Sample a large number of points from the GMM
        2. Compute the log probability of each sample
        3. Take the negative mean of these log probabilities
        
        Args:
        gmm (torch.distributions.MixtureSameFamily): The GMM distribution

        Returns:
        float: Approximated entropy of the GMM

        This is a practical approach (avoids numerical integration)
        """
        
        # Number of samples for Monte Carlo approximation
        num_samples = 10000
        
        # Sample from the GMM
        samples = gmm.sample((num_samples,))
        
        # Compute log probabilities
        log_probs = gmm.log_prob(samples)
        
        # Approximate entropy
        entropy = -log_probs.mean()
        
        return entropy

    def visualize_gmm(self, gmm, num_samples=20000):
        """
        Visualize the GMM distribution using Seaborn and Matplotlib.
        """
        # Set the style for Seaborn
        sns.set_style("whitegrid")
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Sample from the GMM
        samples = gmm.sample((num_samples,))
        print(f"\nsamples: {samples.shape}\n")

        # Ensure samples are 2D
        if samples.dim() == 3:
            samples = samples.squeeze(1)
        
        # Create a scatter plot using Seaborn
        sns.scatterplot(x=samples[:, 0], y=samples[:, 1], alpha=0.5, ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Location", fontsize=12)
        ax.set_ylabel("Thickness", fontsize=12)
        ax.set_title("Gaussian Mixture Model", fontsize=14)
        
        # Adjust x-axis
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks(np.arange(-0.5, 1.5, 0.1))
        ax.set_xticklabels([f"{x:.1f}" for x in np.arange(-0.5, 1.5, 0.1)])
        
        # Adjust y-axis
        ax.set_ylim(self.min_thickness, self.max_thickness)
        y_ticks = np.linspace(self.min_thickness, self.max_thickness, 10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])
        
        # Add a colorbar to represent density
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="YlOrRd", fill=True, cbar=True, ax=ax)
        
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
    
    def act(self, x, edge_index, edge_attr, batch):
        """
        Propose up to max_proposals number of crosswalks.
        For use in policy gradient methods, the log probabilities of the actions are needed.

        We are using reparameterization trick but is that a good assumption for the distribution of the actions?
        problem with normal distribution is that it assumes a single mode. When sampling, likelihood of getting a sample far away from the mean is low (depends on std).
        Instead, we use a mixture of Gaussians. 
            - Can model more complex distributions
            - Can capture multiple modes in the distribution
            - Flexibility: Can be parameterized to have different means and variances for each component

        - Should thickness and location be independent? No. Particular thickness for a specific location is what is needed. 
        - The distribution model should jointly model the two (location and thickness). 
        """

        # Get node embeddings by passing input through GAT layers
        node_embeddings = self.forward(x, edge_index, edge_attr, batch)
        print(f"\ngraph/node embeddings: {node_embeddings}\n")
        #print(f"\nnode_embeddings.mean(dim=0): {node_embeddings.mean(dim=0)}\n") # dim=0 is not required because of readout layer.

        # Predict the number of proposals
        num_proposals_logits = self.num_proposals_layer(node_embeddings) 
        print(f"\nHERE: num_proposals_logits: {num_proposals_logits}\n")
        # Apply temperature to control exploration-exploitation trade-off
        num_proposals_probs = F.softmax(num_proposals_logits / self.temperature, dim=-1)  # Convert to probabilities for each index (total sum to 1) with temperature

        # multinomial distribution is used to model the outcome of selecting one option from a set of mutually exclusive options, where each option has a specific probability of being chosen.
        # Sample the number of proposals (add 1 to ensure at least 1 proposal)
        num_actual_proposals = torch.multinomial(num_proposals_probs, 1).item() + 1 # The inner 1 to the number of draws
        print(f"\nnum_actual_proposals: {num_actual_proposals}\n")

        gmm = self.get_gmm_distribution(node_embeddings)

        # Sample the num_actual_proposals number of proposals directly from the distribution
        samples = gmm.sample((num_actual_proposals,))
        locations, thicknesses = samples.split(1, dim=-1)

        # Normalize locations to be between 0 and 1
        locations = torch.sigmoid(locations)
        
        # Scale thicknesses to be between min_thickness and max_thickness
        # First, sigmoid normalizes to [0, 1], then we scale and shift to [min_thickness, max_thickness]
        thicknesses = self.min_thickness + torch.sigmoid(thicknesses) * (self.max_thickness - self.min_thickness)

        print(f"\nlocations: {locations.shape}\n")
        print(f"\nthicknesses: {thicknesses.shape}\n")

        # Create a padded fixed-sized output. 
        output = torch.full((self.max_proposals, 2), 0.0) # padded with 0s
        output[:num_actual_proposals, 0] = locations.squeeze()
        output[:num_actual_proposals, 1] = thicknesses.squeeze()

        # Compute log probabilities of the chosen actions. Represents "the likelihood of the model choosing each specific proposal (location and thickness) for a crosswalk"
        log_probs = gmm.log_prob(samples)

        # The algorithm expects a single total log probability for the entire batch of actions. 
        total_log_prob = log_probs.sum()

        print(f"\ntotal_log_prob: {total_log_prob}\n")

        return output, num_actual_proposals, total_log_prob

    def evaluate(self, state, actions):
        """
        Evaluate the state-action pair for PPO.
        Returns action log probabilities, state values, and entropy.
        state is the full state (x, edge_index, edge_attr, batch)
        actions is the full action (max_proposals, 2)
        """
        x, edge_index, edge_attr, batch = state

        # Get node embeddings
        node_embeddings = self.forward(x, edge_index, edge_attr, batch)
        print(f"\ngraph/node embeddings: {node_embeddings}\n")
        
        # Predict the number of proposals
        num_proposals_logits = self.num_proposals_layer(node_embeddings)
        print(f"\nHERE:num_proposals_logits: {num_proposals_logits}\n")

        num_proposals_probs = F.softmax(num_proposals_logits / self.temperature, dim=-1)
        
        gmm = self.get_gmm_distribution(node_embeddings)

        # Compute log probabilities of the actions
        action_log_probs = gmm.log_prob(actions)

        # Compute entropy (how uncertain or random the actions selection is)
        entropy = self.gmm_entropy(gmm) # already performs the mean.
        print(f"\nentropy: {entropy}\n")

        # Process actions while preserving structure
        # Actions are locations (in the range [0, 1]) and thicknesses. Should we sort them? 
        print(f"\nHERE: actions: {actions}, flattened: {actions.flatten()}\n")



        action_embedding = self.action_encoder(actions.flatten())
        print(f"\naction_embedding: {action_embedding.shape}\n")
        
        # Concatenate graph embedding with the processed actions
        state_action = torch.cat([node_embeddings.squeeze(0), action_embedding], dim=-1)
        print(f"\nstate_action: {state_action.shape}\n")

        # Compute state value
        state_value = self.critic_layer(state_action)
        print(f"\nstate_value: {state_value.shape}\n")
        
        return action_log_probs, state_value, entropy, num_proposals_probs
    
    def param_count(self):
        """
        Count the total number of parameters in the model.
        """
        # Shared parameters (GATv2Conv layers)
        shared_params = sum(p.numel() for p in self.conv1.parameters()) + \
                        sum(p.numel() for p in self.conv2.parameters())

        # Actor-specific parameters
        actor_params = sum(p.numel() for p in self.num_proposals_layer.parameters())

        # Critic-specific parameters
        critic_params = sum(p.numel() for p in self.critic_layer.parameters()) + \
                        sum(p.numel() for p in self.action_encoder.parameters())

        # GMM parameters (shared between actor and critic)
        gmm_params = sum(p.numel() for p in self.gmm_layers.parameters())

        # Total parameters
        total_params = shared_params + actor_params + critic_params + gmm_params

        return {
            "shared": shared_params,
            "actor_total": shared_params + actor_params + gmm_params,
            "critic_total": shared_params + critic_params + gmm_params,
            "gmm": gmm_params,
            "total": total_params
        }


################ EXAMPLE USAGE #################

# Set up parameters
num_nodes = 50
num_edges = 200
edge_dim = 2  # Number of features per edge

in_channels = 2  # Number of input features per node (e.g., x and y coordinates)
hidden_channels = 8
out_channels = 16

action_hidden_channels = 32
gmm_hidden_dim = 32 

initial_heads = 16 # initial number of attention heads in GAT layers
second_heads = 4 # number of attention heads in the second GAT layer
action_dim = 10  # Maximum number of crosswalk proposals

batch = torch.zeros(num_nodes, dtype=torch.long)

print("Initializing GATv2ActorCritic model...") 
model = GATv2ActorCritic(in_channels=in_channels, 
                         hidden_channels=hidden_channels, 
                         out_channels=out_channels, 
                         initial_heads=initial_heads, 
                         second_heads=second_heads, 
                         edge_dim=edge_dim, 
                         action_hidden_channels=action_hidden_channels,
                         action_dim=action_dim,
                         gmm_hidden_dim=gmm_hidden_dim)

# Generate dummy data
print("\nGenerating dummy input data...")
x = torch.rand((num_nodes, in_channels))  
edge_index = torch.randint(0, num_nodes, (2, num_edges))  
edge_attr = torch.rand((num_edges, edge_dim))  

print(f"Node features shape: {x.shape}")
print(f"Edge index shape: {edge_index.shape}")
print(f"Edge attributes shape: {edge_attr.shape}")

# Print model information
print("\nModel architecture:")
print(model)

print("\nModel parameter count:")
param_count = model.param_count()
for key, value in param_count.items():
    print(f"{key}: {value}")

# Generate crosswalk proposals using the model
print("\nGenerating crosswalk proposals...")
proposed_crosswalks, num_actual_proposals, total_log_prob = model.act(x, edge_index, edge_attr, batch)

print(f"\nNumber of actual proposals: {num_actual_proposals}")
print(f"Total log probability of the action: {total_log_prob.item()}")
print("\nProposed crosswalks:")
for i, (location, thickness) in enumerate(proposed_crosswalks):
    if i < num_actual_proposals:
        print(f"  Proposal {i+1}: Location: {location:.4f}, Thickness: {thickness:.2f}")
    else:
        print(f"  Padding {i+1}: Location: {location:.4f}, Thickness: {thickness:.2f}")

# Evaluate an action
print("\nEvaluating the proposed action...")
state = (x, edge_index, edge_attr, batch)
action = proposed_crosswalks

action_log_probs, state_value, entropy, num_proposals_probs = model.evaluate(state, action)

print(f"Action log probabilities shape: {action_log_probs.shape}")
print(f"State value: {state_value.item()}")
print(f"Entropy: {entropy.item()}")
print(f"Number of proposals probabilities shape: {num_proposals_probs.shape}")

print("\nDetailed evaluation results:")
print(f"  Action log probabilities: {action_log_probs}")
print(f"  State value: {state_value.item()}")
print(f"  Entropy: {entropy.item()}")
print(f"  Number of proposals probabilities: {num_proposals_probs}")

# Visualize the GMM distribution
print("\nVisualizing the GMM distribution...")
gmm = model.get_gmm_distribution(model.forward(x, edge_index, edge_attr, batch))
model.visualize_gmm(gmm)


