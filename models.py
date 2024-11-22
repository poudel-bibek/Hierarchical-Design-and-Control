import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torch.distributions import Categorical, Bernoulli
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical

######## CNN model ########
class CNNActorCritic(nn.Module):
    def __init__(self, in_channels, action_dim, **kwargs):
        """
        CNN Actor-Critic network with configurable size (designed to be compatible with hyper-parameter tuning)
        we are applying conv2d, the state should be 2d with a bunch of channels.
        Choices: 
            Small: 4 Conv layers, 3 Linear layers
            Medium: 6 Conv layers, 5 Linear layers

        Regularization: Dropout and Batch Norm (mitigation of internal covariate shift)
        Conservatively using pooling layers. Every piece of information is important, however we also want to avoid overfitting and keep parameters modest. 
        Dilation: For the first layer, experiment with dilation. (Disabled for now)

        During hyper-param sweep, the model size changes based on one of the dimension of the input (action_duration). 
        Even at high action durations, the model size is around 4.1M parameters. 
        """
        super(CNNActorCritic, self).__init__()
        self.in_channels = in_channels
        self.action_dim = action_dim 
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
            )
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
            )
            hidden_dim = 256

        # Calculate the size of the flattened CNN output
        with torch.no_grad():
            sample_input = torch.zeros(1, self.in_channels, self.action_duration, self.per_timestep_state_dim) # E.g., (1,1,10,74) batch size of 1, 1 channel, 10 timesteps, 74 state dims
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
            nn.Linear(hidden_dim // 2, self.action_dim)
        )
        
        # Critic-specific layers
        self.critic_layers = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def actor(self, state,):
        shared_features = self.shared_cnn(state)
        action_logits = self.actor_layers(shared_features)
        print(f"\n\nAction logits: {action_logits}\n\n")
        return action_logits
    
    def critic(self, state):
        shared_features = self.shared_cnn(state)
        return self.critic_layers(shared_features)
    
    def act(self, state):
        """
        Select an action based on the current state:
        - First action: 4-class classification for traffic light
        - Second and third actions: binary choices for crosswalks
        """
        state_tensor = state.reshape(1, self.in_channels, self.action_duration, self.per_timestep_state_dim)
        action_logits = self.actor(state_tensor)
        print(f"\nAction logits: {action_logits}")
        
        # Split logits into traffic light and crosswalk decisions
        traffic_logits = action_logits[:, :4]  # First 4 logits for traffic light (4-class)
        crosswalk_logits = action_logits[:, 4:]  # Last 2 logits for crosswalks (binary)
        print(f"\nTraffic logits: {traffic_logits}")
        print(f"Crosswalk logits: {crosswalk_logits}")
        
        # Multi-class classification for traffic light
        traffic_probs = F.softmax(traffic_logits, dim=1)
        traffic_dist = Categorical(traffic_probs)
        traffic_action = traffic_dist.sample() # This predicts 0, 1, 2, or 3
        print(f"\nTraffic probabilities: {traffic_probs}")
        print(f"Traffic action: {traffic_action}")
        
        # Binary choices for crosswalks
        crosswalk_probs = torch.sigmoid(crosswalk_logits)
        crosswalk_dist = Bernoulli(crosswalk_probs)
        crosswalk_actions = crosswalk_dist.sample() # This predicts 0 or 1
        print(f"\nCrosswalk probabilities: {crosswalk_probs}")
        print(f"Crosswalk actions: {crosswalk_actions}\n")
        
        # Combine actions
        combined_action = torch.cat([traffic_action, crosswalk_actions.squeeze(0)], dim=0)
        print(f"\nCombined action: {combined_action}")
        
        # Calculate log probabilities
        log_prob = traffic_dist.log_prob(traffic_action) + crosswalk_dist.log_prob(crosswalk_actions).sum()
        print(f"\nLog probability: {log_prob}")
        
        return combined_action.long(), log_prob

    def evaluate(self, states, actions):
        """
        Evaluates a batch of states and actions.
        States are passed to actor to get action logits, using which we get the probabilities and then the distribution. similar to act function.
        Then using the sampled actions, we get the log probabilities and the entropy. 
        Finally, we pass the states to critic to get the state values. (used to compute the value function component of the PPO loss)
        The entropy is used as a regularization term to encourage exploration.
        """
        action_logits = self.actor(states)
        
        # Split logits and actions
        traffic_logits = action_logits[:, 0:2]
        crosswalk_logits = action_logits[:, 2:]
        traffic_actions = actions[:, 0:2].argmax(dim=1)  # Convert one-hot back to index
        crosswalk_actions = actions[:, 2:]
        
        # Evaluate traffic direction actions
        traffic_probs = F.softmax(traffic_logits, dim=1)
        #TODO:Visualize this?
        print(f"\nTraffic probabilities: {traffic_probs}\n")

        traffic_dist = Categorical(traffic_probs)
        traffic_log_probs = traffic_dist.log_prob(traffic_actions)
        
        # Evaluate crosswalk actions
        crosswalk_probs = torch.sigmoid(crosswalk_logits)
        crosswalk_dist = Bernoulli(crosswalk_probs)
        crosswalk_log_probs = crosswalk_dist.log_prob(crosswalk_actions)
        
        # Combine log probabilities
        action_log_probs = traffic_log_probs + crosswalk_log_probs.sum(dim=1)
        
        # Calculate entropy 
        dist_entropy = traffic_dist.entropy() + crosswalk_dist.entropy().sum(dim=1)
        
        state_values = self.critic(states)
        
        return action_log_probs, state_values, dist_entropy

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
class GATv2ActorCritic(nn.Module):
    """
    GATv2 with edge features.
    """

    def __init__(self, in_channels, 
                 action_dim,
                 hidden_channels = None, 
                 out_channels = None, 
                 initial_heads = None, 
                 second_heads = None, 
                 edge_dim = None, 
                 action_hidden_channels = None, 
                 gmm_hidden_dim = None, 
                 num_mixtures = 3, 
                 actions_per_node=2, 
                 dropout_rate=0.2, 
                 min_thickness=0.1, 
                 max_thickness=10.0):
        
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
        # At every timestep, the actions is a whole bunch of things of max size. Critic has to evaluate all that (insted of just the relevant parts).
        # Thickness and location values that are not in the proposal are set to -1 (which is close to minimum of 0.1)
        # model could potentially interpret these as meaningful values. 

        """
        super(GATv2ActorCritic, self).__init__()
        self.max_proposals = action_dim
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.num_mixtures = num_mixtures
        self.dropout_rate = dropout_rate
        self.elu = nn.ELU()

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
        
        # Temperature parameter (of the softmax function) for controlling exploration in action selection
        # A lower temperature (0.1) makes the distribution more peaked (more deterministic), while a higher temperature (2.0) makes it more uniform (more random).
        #self.temperature = nn.Parameter(torch.ones(1)) # this is a learnable parameter. No need for this to be a learnable. Other mechanisms to control exploration.
        self.temperature = 1.0 # fixed temperature

        # Finally. After the readout layer (upto that, things are shared), the output is passed to either an actor or a critic.
        # Sequential layers for actor. Actor predicts GMM parameters and the number of times to sample from the GMM.
        # Stacked linear layers for GMM parameters for joint prediction of all GMM parameters (instead of separate layers for each)
        # Output: num_mixtures * 5 values
        #   - num_mixtures for mix logits (weights of each Gaussian), determines the weight of this Gaussian in the mixture
        #   - num_mixtures * 2 for means (location and thickness), determines the center of the Gaussian 
        #   - num_mixtures * 2 for covariances (diagonal, for simplicity), determines the spread of the Gaussian

        self.actor_gmm_layers = nn.Sequential(
            nn.Linear(out_channels * second_heads, gmm_hidden_dim),
            nn.ReLU(),
            nn.Linear(gmm_hidden_dim, num_mixtures * 5)  # 5 = 1 (mix_logit) + 2 (means) + 2 (covs)
        )

        # Linear layer for predicting the number of proposals
        self.actor_num_proposals_layer = torch.nn.Linear(out_channels * second_heads, action_dim)

        # Sequential layers for critic
        # This layer gets input the graph embedding and the action embedding. 
        self.critic_layers = nn.Sequential(
            # graph/ node embedding output is shaped (out_channels * second_heads) (1D output of the readout layer)
            nn.Linear(out_channels * second_heads, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # output a single value
        )

    def readout_layer(self, x, batch):
        """
        As a number of approaches are possible, this is a separate function.
        """
        # global_mean_pool to average across the nodes for each graph in a batch.
        return global_mean_pool(x, batch)

    def actor(self, state_batch):
        """
        Forward pass: consists of two parts (all in one head)
        - GMM parameters prediction 
        - Number of proposals prediction (# of times to sample from GMM)

        State = Data or Batch object with 4 tensors:
        - node features (x) = (num_nodes, in_channels)
        - edge index (edge_index) = (2, num_edges) connections between nodes. # e.g., edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
        - edge attributes (edge_attr) = Edge features (num_edges, edge_dim)
        - batch (batch) = 
        """
        y = self.elu(self.conv1(state_batch.x, state_batch.edge_index, state_batch.edge_attr))
        y = F.dropout(y, p=self.dropout_rate, training=self.training)

        # The same edge_attr is used for both the first and second GAT layers. 
        # Apply second GAT layer with edge features (no activation here as it's the final layer)
        # Why are edge features passed again?
        # Preserving edge information: By passing the edge attributes to each layer, the model can maintain information about the edge features throughout the network depth. 
        # Different learned attention: Each GAT layer learns its own attention mechanism. By providing the edge features to each layer, you allow each layer to learn how to use these features differently
        # Residual-like connections for edges: In a sense, passing edge features to each layer creates a form of residual connection for edge information.         
        y = self.conv2(y, state_batch.edge_index, state_batch.edge_attr)
        y = self.readout_layer(y, state_batch.batch)

        gmm_params = self.actor_gmm_layers(y)
        num_proposals_logits = self.actor_num_proposals_layer(y)

        return gmm_params, num_proposals_logits

    def critic(self, state_batch):
        """
        Critic forward pass.
        """

        y = self.elu(self.conv1(state_batch.x, state_batch.edge_index, state_batch.edge_attr))
        y = F.dropout(y, p=self.dropout_rate, training=self.training)
        y = self.conv2(y, state_batch.edge_index, state_batch.edge_attr)
        y = self.readout_layer(y, state_batch.batch)

        return self.critic_layers(y).squeeze(-1)  # Ensure output is of shape (batch_size,)
    
            # For testing: duplicate data 3 times to create multiple batches
        # batch_size = 3
        # num_nodes = state_batch['x'].size(0)
        # batch = torch.cat([
        #     torch.zeros(num_nodes, dtype=torch.long), # 0 
        #     torch.ones(num_nodes, dtype=torch.long), # 1
        #     2 * torch.ones(num_nodes, dtype=torch.long) # 2
        # ])
        
        # # Duplicate node features, edge indices and edge attributes 3 times
        # state_batch['x'] = state_batch['x'].repeat(3,1)
        
        # # For edge_index, need to offset node indices for each batch
        # edge_index_offset1 = state_batch['edge_index'] + num_nodes
        # edge_index_offset2 = state_batch['edge_index'] + (2 * num_nodes)
        # state_batch['edge_index'] = torch.cat([
        #     state_batch['edge_index'],
        #     edge_index_offset1,
        #     edge_index_offset2
        # ], dim=1)
        
        # state_batch['edge_attr'] = state_batch['edge_attr'].repeat(3,1)
        # state_batch['batch_size'] = 3  # Update batch size
        # device = next(self.parameters()).device
        ### end testing

    def get_gmm_distribution(self, state_batch):
        """
        Returns:
        - gmm (MixtureSameFamily): The GMM distribution.
        - num_proposals_probs (Tensor): Probabilities for the number of proposals.
        """
        device = next(self.parameters()).device

        # From given batch_size, make GAT batch.
        # A GAT batch with batch_size = 1 looks like [0, 0, 0, 0, ... num_nodes times]
        # A GAT batch with batch_size = 4 looks like [0, 0, 1, 1, 2, 2, 3, 3, ... num_nodes times]
        # GAT batch (required when graph passes through GAT policy) is different from gradient mini-batch.
        batch_size = state_batch.num_graphs  # Get number of graphs in the batch
        state_batch = state_batch.to(device)

        # actor returns stuff for entire batch. (batchsize, num_mixtures * 5) and (batchsize, max_proposals)
        gmm_params, num_proposals_logits = self.actor(state_batch)
        
        # Apply temperature to control exploration-exploitation
        num_proposals_probs_batch = F.softmax(num_proposals_logits / self.temperature, dim=-1)  # Convert to probabilities for each index (total sum to 1) with temperature

        # Split parameters for each batch element
        mix_logits, means, covs = gmm_params.split([self.num_mixtures, self.num_mixtures * 2, self.num_mixtures * 2], dim=-1)
        means = means.view(batch_size, self.num_mixtures, 2)
        covs = F.softplus(covs).view(batch_size, self.num_mixtures, 2) # Ensure positive covariance
        
        # Create GMM distribution for each batch element
        gmm_batch = []
        for b in range(batch_size):
            mix = Categorical(logits=mix_logits[b]) # Categorical distribution for the mixture probabilities
            covariance_matrices = torch.diag_embed(covs[b]) # Create diagonal covariance matrices
            comp = MultivariateNormal(means[b], covariance_matrices) # Multivariate normal distributions for each component
            gmm = MixtureSameFamily(mix, comp) # Mixture of Gaussians distribution
            gmm_batch.append(gmm)

        return gmm_batch, num_proposals_probs_batch
    
    def act(self, state_batch, iteration=None, visualize=False):
        """
        Sample actions from the policy given the state (propose upto max_proposals number of crosswalks).
        For use in policy gradient methods, the log probabilities of the actions are needed.

        Using reparameterization trick (assumes that actions follow a certain continuous and differentiable distribution)
        Why not the default normal distribution: it assumes a single mode i.e., when sampling, likelihood of getting a sample far away from the mean is low (depends on std).
        Instead, we use a mixture of Gaussians. 
            - Can model more complex distributions
            - Can capture multiple modes in the distribution
            - Flexibility: Can be parameterized to have different means and variances for each component

        Should thickness and location be independent? No. Particular thickness for a specific location is what is needed. 
        Hence, the distribution jointly models the two (location and thickness). 

        multinomial distribution is used to model the outcome of selecting one option from a set of mutually exclusive options, where each option has a specific probability of being chosen.
        """

        # If a single instance is passed, wrap it around a list and make a batch.
        if isinstance(state_batch, Data):
            state_batch = Batch.from_data_list([state_batch])

        batch_size = state_batch.num_graphs 

        # Get GMM parameters and number of proposals distributions
        gmm_batch, num_proposals_probs_batch = self.get_gmm_distribution(state_batch)
        
        # This visualization is only meaningful during act (i.e., when single graph is passed)
        if visualize and iteration is not None:
            self.visualize_gmm(gmm_batch[0], iteration=iteration)
        
        # Sample number of proposals for each batch element (add 1 to ensure at least 1 proposal)
        num_actual_proposals = torch.multinomial(num_proposals_probs_batch, 1).squeeze(-1) + 1
        
        # Initialize output tensor for all batches (2 because location and thickness)
        output = torch.full((batch_size, self.max_proposals, 2), 0.0)
        total_log_probs = torch.zeros(batch_size)
        
        # Process each batch element
        for b in range(batch_size):
            # Sample proposals for this batch element
            n_proposals = num_actual_proposals[b].item()
            samples = gmm_list[b].sample((n_proposals,))
            
            # Split into locations and thicknesses
            locations, thicknesses = samples.split(1, dim=-1)
            
            # Normalize locations to be between 0 and 1
            locations = torch.sigmoid(locations)
            
            # Scale thicknesses to be between min_thickness and max_thickness
            # First, sigmoid normalizes to [0, 1], then we scale and shift to [min_thickness, max_thickness]
            thicknesses = self.min_thickness + torch.sigmoid(thicknesses) * (self.max_thickness - self.min_thickness)
            
            # Store in output tensor
            output[b, :n_proposals, 0] = locations.squeeze()
            output[b, :n_proposals, 1] = thicknesses.squeeze()
            
            # Compute log probabilities for this batch element
            log_probs = gmm_list[b].log_prob(samples)
            total_log_probs[b] = log_probs.sum()
        
        return output, num_actual_proposals, total_log_probs
    
    def evaluate(self, states_batch, actions_batch):
        """
        Returns:
            action_log_probs (Tensor): Log probabilities of the actions.
            state_values (Tensor): Values of the states.
            entropy (Tensor): Entropy of the policy.
        """
        batch_size = actions_batch.size(0)
        
        # Get distributions
        gmm_list, num_proposals_probs = self.get_gmm_distribution(states_batch)
        
        # Initialize return tensors
        action_log_probs = torch.zeros(batch_size, device=actions_batch.device)
        entropy = torch.zeros(batch_size, device=actions_batch.device)
        
        # Process each batch element
        for b in range(batch_size):
            # Get actual proposals for this batch element
            n_proposals = num_proposals_batch[b].item()
            actual_actions = actions_batch[b, :n_proposals]
            
            # Compute log probabilities and entropy for this batch element
            action_log_probs[b] = gmm_list[b].log_prob(actual_actions).sum()
            entropy[b] = self.gmm_entropy(gmm_list[b])
        
        # Get state values
        state_values = self.critic(states_batch)
        
        return action_log_probs, state_values, entropy, num_proposals_probs
    
    def gmm_entropy(self, gmm_single):
        """
        Entropy measures the uncertainty or randomness in the action selection process given a state.
        - High Entropy: policy is highly uncertain about which action to take (i.e., it assigns similar probabilities to multiple actions), 
          this encourages exploration because the policy is not overly confident in selecting a single action.
        - Low Entropy: policy is very certain about which action to take (i.e., it assigns a high probability to a specific action and low probabilities to others), 
          which indicates more deterministic behavior.

        For a GMM, the entropy does not have a closed-form solution.
        Approximating it using Monte Carlo sampling (practical approach that avoids numerical integration)
        1. Sample a large number of points from the GMM
        2. Compute the log probability of each sample
        3. Take the negative mean of these log probabilities
        
        Returns: Approximated entropy of the GMM
        """
        num_samples = 10000
        samples = gmm_single.sample((num_samples,))
        log_probs = gmm_single.log_prob(samples)
        entropy = -log_probs.mean() 
        return entropy
    
    def param_count(self):
        """
        Count the total number of parameters in the model.
        """
        # Shared params (GATv2Conv layers)
        shared_params = sum(p.numel() for p in self.conv1.parameters()) + \
                        sum(p.numel() for p in self.conv2.parameters())

        # Actor-specific 
        actor_params = sum(p.numel() for p in self.actor_gmm_layers.parameters()) + \
                        sum(p.numel() for p in self.actor_num_proposals_layer.parameters())

        # Critic-specific 
        critic_params = sum(p.numel() for p in self.critic_layers.parameters())

        total_params = shared_params + actor_params + critic_params 
        return {
            "shared": shared_params,
            "actor_total": shared_params + actor_params,
            "critic_total": shared_params + critic_params,
            "total": total_params}
    
    def visualize_gmm(self, gmm_single, num_samples=20000, locations=None, iteration=None):
        """
        Visualize the GMM distribution in 3D.
        If locations are provided, they are marked as red crosses.

        Args:
            gmm_single (MixtureSameFamily): The GMM distribution for a single batch element.
            num_samples (int): Number of samples to generate for visualization.
            locations (Tensor or ndarray): Locations to mark on the plot, shape (N, 2).
        """
        

        save_path = f"gmm_iterations/gmm_distribution_iteration_{iteration}.png"
        # Sample from the GMM
        samples = gmm_single.sample((num_samples,))  # Shape: (num_samples, 2)
        samples = samples.detach().cpu().numpy()

        # Create a grid for plotting
        x = samples[:, 0]  # Location dimension
        y = samples[:, 1]  # Thickness dimension

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        X = np.linspace(xmin, xmax, 100)
        Y = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(X, Y)

        # Create positions array for GMM evaluation
        positions = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                               dtype=torch.float32,
                               device=gmm_single.component_distribution.loc.device)  # Get device from component distribution

        # Evaluate the GMM on the grid and move to CPU for numpy conversion
        Z = gmm_single.log_prob(positions).detach().cpu()
        Z = np.exp(Z.numpy()).reshape(X.shape)

        fig = plt.figure(figsize=(10, 8), dpi=300)   
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.7)

        # Plot the provided locations as red crosses
        if locations is not None:
            if isinstance(locations, torch.Tensor):
                locations = locations.detach().cpu().numpy()
            if locations.ndim == 1:
                locations = locations.reshape(-1, 2)
            
            z_vals = interpolate.griddata((X.ravel(), Y.ravel()), Z.ravel(), 
                                        (locations[:, 0], locations[:, 1]), method='linear')
            ax.scatter(locations[:, 0], locations[:, 1], zs=z_vals, 
                      c='r', marker='x', s=100, label='Given Locations')

        ax.set_xlabel('Location', fontweight='bold')
        ax.set_ylabel('Thickness', fontweight='bold')
        ax.set_zlabel('Density', fontweight='bold')
        ax.set_title('3D GMM', fontweight='bold')
        if locations is not None:
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"GMM distribution plot saved to {save_path}")
        # plt.show()
        plt.close()


################ EXAMPLE USAGE #################
"""

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

"""