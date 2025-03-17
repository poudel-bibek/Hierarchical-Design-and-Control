import torch
import numpy as np
import torch.nn as nn
from matplotlib import cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from entropy_calc import gmm_entropy_monte_carlo, gmm_entropy_legendre
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical, Bernoulli

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogoal initialization of weights and Constant initialization of biases.
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP_ActorCritic(nn.Module):
    def __init__(self, in_channels, action_dim, **kwargs):
        """
        MLP Actor-Critic network in two sizes: small, medium. 
        - Two separate networks for actor and critic (No shared backbone as there is no feature extraction).
        - Expects inputs of shape (B, in_channels, action_duration, per_timestep_state_dim) then flattens to (B, -1).
        """
        super(MLP_ActorCritic, self).__init__()
        in_channels = in_channels
        action_duration = kwargs.get('action_duration')
        per_timestep_state_dim = kwargs.get('per_timestep_state_dim')

        with torch.no_grad():
            sample_input = torch.zeros(1, in_channels, action_duration, per_timestep_state_dim)
            self.input_dim = sample_input.numel()  # total number of features, e.g. 1 * c * d * s

        if kwargs.get('activation') == "tanh":
            activation = nn.Tanh
        elif kwargs.get('activation') == "relu":
            activation = nn.ReLU
        elif kwargs.get('activation') == "leakyrelu":
            activation = nn.LeakyReLU
        # dropout_rate = kwargs.get('dropout_rate')

        model_size = kwargs.get('model_size')
        if model_size == 'small':
            actor_hidden_sizes = [256, 128, 64, 32]
            critic_hidden_sizes = [256, 128, 64, 32]
        elif model_size == 'medium':
            actor_hidden_sizes = [512, 256, 128, 64, 32]
            critic_hidden_sizes = [512, 256, 128, 64, 32]
        
        # actor
        actor_layers = []
        input_size_actor = self.input_dim
        for h in actor_hidden_sizes:
            actor_layers.append(layer_init(nn.Linear(input_size_actor, h)))
            actor_layers.append(nn.LayerNorm(h))  # Add LayerNorm after linear layer
            actor_layers.append(activation())
            # actor_layers.append(nn.Dropout(dropout_rate)) # Disabled for now
            input_size_actor = h
        self.actor_layers = nn.Sequential(*actor_layers)
        self.actor_logits = layer_init(nn.Linear(input_size_actor, action_dim)) # Last layer, no activation

        # critic 
        critic_layers = []
        input_size_critic = self.input_dim
        for h in critic_hidden_sizes:
            critic_layers.append(layer_init(nn.Linear(input_size_critic, h)))
            critic_layers.append(nn.LayerNorm(h))  # Add LayerNorm after linear layer
            critic_layers.append(activation())
            # critic_layers.append(nn.Dropout(dropout_rate)) # Disabled for now
            input_size_critic = h
        self.critic_layers = nn.Sequential(*critic_layers)
        self.critic_value = layer_init(nn.Linear(input_size_critic, 1)) # Last layer, no activation

    def actor(self, state):
        """
        First Flatten the input from 4D (B, C, D, S) to 2D (B, -1)
        Returns the raw action logits from the actor head.
        """
        bsz = state.size(0)
        flat = state.view(bsz, -1)  # shape: (B, in_channels*action_duration*per_timestep_state_dim)
        return self.actor_logits(self.actor_layers(flat))

    def critic(self, state):
        """
        First Flatten the input from 4D (B, C, D, S) to 2D (B, -1)
        Returns the scalar state-value V(s).
        """
        bsz = state.size(0)
        flat = state.view(bsz, -1)  # shape: (B, in_channels*action_duration*per_timestep_state_dim)
        return self.critic_value(self.critic_layers(flat))

    def act(self, state):
        """
        Sample an action exactly like in the CNN version:
          - intersection action from first 4 logits (Categorical)
          - midblock from next 7 logits (Bernoulli)
        """
        # print("Sampling...")
        state = state.reshape(1, 1, state.shape[0], state.shape[1])
        action_logits = self.actor(state)

        # The first 4 logits => intersection (Categorical)
        intersection_logits = action_logits[:, :4]
        # intersection_probs = torch.softmax(intersection_logits, dim=1)
        intersection_dist = Categorical(logits=intersection_logits)
        intersection_action = intersection_dist.sample()  # [1]


        # The next 7 logits => midblock (Bernoulli)
        midblock_logits = action_logits[:, 4:]
        # midblock_probs = torch.sigmoid(midblock_logits)
        midblock_dist = Bernoulli(logits=midblock_logits)
        midblock_actions = midblock_dist.sample()  # shape [1,7]

        # print(f"\nIntersection logits: {intersection_logits}")
        # print(f"\nMidblock logits: {midblock_logits}")

        combined_action = torch.cat([intersection_action, midblock_actions.squeeze(0)], dim=0)
        log_prob = intersection_dist.log_prob(intersection_action) + \
                   midblock_dist.log_prob(midblock_actions).sum()

        # print(f"\nAction Log probability: {log_prob}, shape: {log_prob.shape}")
        return combined_action.int(), log_prob


    def evaluate(self, states, actions):
        """
        Evaluate a batch of states and pre-sampled actions. Same logic as the CNN version.
        """
        # print("Evaluating...")
        action_logits = self.actor(states)
        intersection_logits = action_logits[:, :4]
        midblock_logits = action_logits[:, 4:]

        # Distributions
        # intersection_probs = torch.softmax(intersection_logits, dim=1)
        intersection_dist = Categorical(logits=intersection_logits)
        # midblock_probs = torch.sigmoid(midblock_logits)
        midblock_dist = Bernoulli(logits=midblock_logits)

        # Actions in shape (B,1) for intersection, (B,7) for midblock
        intersection_action = actions[:, :1].squeeze(1).long() # Categorical expects long
        midblock_actions = actions[:, 1:].float()

        intersection_log_probs = intersection_dist.log_prob(intersection_action)
        # print(f"\nIntersection log probs: {intersection_log_probs}, shape: {intersection_log_probs.shape}")
        midblock_log_probs = midblock_dist.log_prob(midblock_actions)
        # print(f"\nMidblock log probs: {midblock_log_probs}, shape: {midblock_log_probs.shape}")
        action_log_probs = intersection_log_probs + midblock_log_probs.sum(dim=1)
        # print(f"\nAction log probs: {action_log_probs}, shape: {action_log_probs.shape}")

        # Entropies
        # print(f"Entropies: intersection: {intersection_dist.entropy()}, midblock: {midblock_dist.entropy().sum(dim=1)}")
        total_entropy = intersection_dist.entropy() + midblock_dist.entropy().sum(dim=1)
        # print(f"Total entropy: {total_entropy}, shape: {total_entropy.shape}")

        # Critic value
        state_values = self.critic(states)
        return action_log_probs, state_values, total_entropy

    def param_count(self):
        """
        Return a dict describing the parameter counts, mirroring the CNN version.
        """
        actor_params = sum(p.numel() for p in self.actor_layers.parameters()) + sum(p.numel() for p in self.actor_logits.parameters())
        critic_params = sum(p.numel() for p in self.critic_layers.parameters()) + sum(p.numel() for p in self.critic_value.parameters())

        return {
            "Actor": actor_params,
            "Critic": critic_params,
            "Total": actor_params + critic_params,
        }

class GAT_v2_ActorCritic(nn.Module):
    """
    GATv2 with edge features.

    Learnings from Exploration paper:
    * To use either log-std or full covariance matrix:

      - Full Covariance:
      - Requires d(d+1)/2 parameters for d components.
      - Allows for correlation between variables (does not have to be diagonal).

      - log-std:  
      - Implicitly uses diagonal covariance matrix.
      - Requires d parameters for d components.
      - Anisotropic (assumption of independence/ variables are uncorrelated).
      - This option chosen for its simplicity.

    * Estimation of Entropy:
      - Discard monte carlo for its non-differentiability.
      - Use Legendre Quadrature for entropy estimation and use monte carlo to validate.

    * Sampling:
      - Make use of implicit stochasticity in sampling from GMM during training (instead of explicit greedy vs random sampling).
      - Greedy sampling from the GMM at evaluation (Maximum propability point corresponding to mean/ center of each component).
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
        super(GAT_v2_ActorCritic, self).__init__()
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

    def actor(self, states_batch):
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
        y = self.elu(self.conv1(states_batch.x, states_batch.edge_index, states_batch.edge_attr))
        y = F.dropout(y, p=self.dropout_rate, training=self.training)

        # The same edge_attr is used for both the first and second GAT layers. 
        # Apply second GAT layer with edge features (no activation here as it's the final layer)
        # Why are edge features passed again?
        # Preserving edge information: By passing the edge attributes to each layer, the model can maintain information about the edge features throughout the network depth. 
        # Different learned attention: Each GAT layer learns its own attention mechanism. By providing the edge features to each layer, you allow each layer to learn how to use these features differently
        # Residual-like connections for edges: In a sense, passing edge features to each layer creates a form of residual connection for edge information.         
        y = self.conv2(y, states_batch.edge_index, states_batch.edge_attr)
        y = self.readout_layer(y, states_batch.batch)

        gmm_params = self.actor_gmm_layers(y)
        num_proposals_logits = self.actor_num_proposals_layer(y)

        return gmm_params, num_proposals_logits

    def critic(self, states_batch):
        """
        Critic forward pass.
        """

        y = self.elu(self.conv1(states_batch.x, states_batch.edge_index, states_batch.edge_attr))
        y = F.dropout(y, p=self.dropout_rate, training=self.training)
        y = self.conv2(y, states_batch.edge_index, states_batch.edge_attr)
        y = self.readout_layer(y, states_batch.batch)

        return self.critic_layers(y).squeeze(-1)  # Ensure output is of shape (batch_size,)

    def get_gmm_distribution(self, states_batch):
        """
        For a GMM with M components, we need: 
        - M mixture weights (logits -> convert to probabilities that sum to 1)
        - M mean vectors (2D: location and thickness)
        - M log-std vectors (2D: log-std in location and thickness)

        Returns:
        - GMM distribution (MixtureSameFamily)
        - num_proposals_probs (Tensor): Probabilities for the number of proposals.
        """
        
        # From given batch_size, make GAT batch.
        # A GAT batch with batch_size = 1 looks like [0, 0, 0, 0, ... num_nodes times]
        # A GAT batch with batch_size = 4 looks like [0, 0, 1, 1, 2, 2, 3, 3, ... num_nodes times]
        # GAT batch (required when graph passes through GAT policy) is different from gradient mini-batch.
        batch_size = states_batch.num_graphs  # Get number of graphs in the batch
        print(f"\n\nHERE: Batch size: {batch_size}\n\n")

        # actor returns stuff for entire batch. (batchsize, num_mixtures * 5) and (batchsize, max_proposals)
        gmm_params, num_proposals_logits = self.actor(states_batch)
        
        # Apply temperature to control exploration-exploitation
        num_proposals_probs_batch = F.softmax(num_proposals_logits / self.temperature, dim=-1)  # Convert to probabilities for each index (total sum to 1) with temperature

        print(f"\n\nGMM params: {gmm_params}\n\n")
        # Split parameters for each batch element
        mix_logits, means, covs = gmm_params.split([self.num_mixtures, self.num_mixtures * 2, self.num_mixtures * 2], dim=-1)
        means = means.view(batch_size, self.num_mixtures, 2)

        # The dimensions of means and covariances are (batch_size, num_mixtures, 2)
        print(f"\n\nBefore transformation: Means: {means}\n Covariances: {covs}\n")
        # Transform to correct ranges here (instead of applying transformations after getting the gmm distribution or after sampling)
        # First component (location) -> [0, 1]. Second component (thickness) -> [min_thickness, max_thickness]
        # Using sigmoid in both cases. TODO: Is a sigmoid normalization what we want?
        means = means.clone() # Create a new tensor instead of modifying in-place
        means[:, :, 0] = torch.sigmoid(means[:, :, 0].clone()) # ... means match all leading dimensions and select 0 from last dimension
        means[:, :, 1] = self.min_thickness + torch.sigmoid(means[:, :, 1].clone()) * (self.max_thickness - self.min_thickness)

        covs = F.softplus(covs).view(batch_size, self.num_mixtures, 2) # Ensure positive covariance
        # similarly for covariances. Covariance scaling affects how spread out or peaked the distribution is.
        scaling_factor = 1 / 32
        # Scale location covariance to be proportional to [0, 1] range
        covs = covs.clone() # Create a new tensor instead of modifying in-place
        covs[:, :, 0] = covs[:, :, 0] * scaling_factor # location range not squared because its implicitly handeled (1-0)² = 1

        # Scale thickness covariance to be proportional to thickness range. Covariance matrices deal with squared deviations from the mean
        covs[:, :, 1] = covs[:, :, 1] * (self.max_thickness - self.min_thickness) ** 2 * scaling_factor
        print(f"\n\nAfter transformation: Means: {means}\n Covariances: {covs}\n")

        # Create GMM distribution for each batch element
        gmm_batch = []
        for b in range(batch_size):
            mix = Categorical(logits=mix_logits[b]) # Categorical distribution for the mixture probabilities
            covariance_matrices = torch.diag_embed(covs[b]) # Create diagonal covariance matrices
            comp = MultivariateNormal(means[b], covariance_matrices) # Multivariate normal distributions for each component
            gmm = MixtureSameFamily(mix, comp) # Mixture of Gaussians distribution
            gmm_batch.append(gmm)

        return gmm_batch, num_proposals_probs_batch
    
    def act(self, states_batch, iteration=None, visualize=False):
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

        # If a single instance is passed (happens only in act), wrap it around a list and make a batch.
        if isinstance(states_batch, Data):
            states_batch = Batch.from_data_list([states_batch])

        batch_size = states_batch.num_graphs 
        device = next(self.parameters()).device
        print(f"\n\nState batch size: {states_batch.size()}\n\n")
        # Get GMM parameters and number of proposals distributions
        gmm_batch, num_proposals_probs_batch = self.get_gmm_distribution(states_batch.to(device))
        # MixtureSameFamily combines Categorical distribution with torch.Size([3]) for the mixture weights
        # Categorical Distribution: Determines the probability of selecting each component (mixing weights). In this case, we have 3 weights that sum to 1.
        # MultivariateNormal distribution with: loc (means) of size torch.Size([3, 2]) and covariance_matrix of size torch.Size([3, 2, 2])
        print("\n\nGMM Distribution Details:")
        for i, gmm in enumerate(gmm_batch):
            print(f"\nBatch element {i}:")
            print(f"Mixture weights: {torch.exp(gmm.mixture_distribution.logits)}")  # Convert logits to probabilities
            print(f"Component means: {gmm.component_distribution.loc}")
            print(f"Component covariances:\n{gmm.component_distribution.covariance_matrix}\n")

        # Sample number of proposals for each batch element (add 1 to ensure at least 1 proposal)
        num_actual_proposals = torch.multinomial(num_proposals_probs_batch, 1).squeeze(-1) + 1
        print(f"\n\nnum_actual_proposals: {num_actual_proposals.shape, num_actual_proposals}\n\n")

        # Initialize output tensors (2 because location and thickness)
        proposals = torch.full((batch_size, self.max_proposals, 2), -1.0, dtype=torch.float32, device=device) # Initialize with -1 so that its easier to infer the actual proposals in critic without passing them around.
        log_probs = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):

            # Sample proposals for this batch element
            samples = gmm_batch[b].sample((num_actual_proposals[b].item(),))
            locations, thicknesses = samples.split(1, dim=-1)
            
            # Clamp the locations to [0,1]
            locations = torch.clamp(locations, 0.0, 1.0)
            thicknesses = torch.clamp(thicknesses, self.min_thickness, self.max_thickness)
            
            # Recombine the samples
            samples = torch.cat([locations, thicknesses], dim=-1)
            
            # Visualization is only meaningful during act (i.e., not during evaluation)
            if visualize and iteration is not None:
                markers = (locations.squeeze().detach().cpu().numpy(), thicknesses.squeeze().detach().cpu().numpy())
                self.visualize_gmm(gmm_batch[b], markers=markers, batch_index=b, thickness_range=(self.min_thickness, self.max_thickness), location_range=(0, 1), iteration=iteration)

            # Store in output tensor
            proposals[b, :num_actual_proposals[b], 0] = locations.squeeze()
            proposals[b, :num_actual_proposals[b], 1] = thicknesses.squeeze()
            
            # Compute log probabilities for this batch element
            log_probs[b] = gmm_batch[b].log_prob(samples).sum()

        return proposals, num_actual_proposals, log_probs
    
    def evaluate(self, states_batch, actions_batch):
        """
        Args:
            states_batch (Batch): Batch of states, each state a Data object.
            actions_batch (Tensor): Batch of actions [batch_size, max_proposals, 2]. 
                - Not all the actions contain actual proposals. Each element in the batch can have a different number of proposals.

        Returns:
            action_log_probs (Tensor): Log probabilities of the actions.
            state_values (Tensor): Values of the states.
            entropy (Tensor): Entropy of the policy.
        """
        
        batch_size = states_batch.num_graphs 
        device = next(self.parameters()).device

        state_values_batch = self.critic(states_batch.to(device))
        print(f"\n\nStates batch size: {states_batch.size()}\n\n")
        # Get distribution (we dont need the samples in critic, which may be changing a lot every time we sample.)
        gmm_batch, _ = self.get_gmm_distribution(states_batch.to(device))
        
        # Initialize return tensors
        action_log_probs = torch.zeros(batch_size, device=device)
        entropy = torch.zeros(batch_size, device=device)
        
        print(f"\n\nActions batch: {actions_batch}\n\n")
        # Compute num_proposals_batch by checking for -1 in actions
        num_proposals_batch = (actions_batch[:, :, 0] != -1).sum(dim=1)
        print(f"\n\nNum proposals batch: {num_proposals_batch}\n\n")

        # Process each batch element
        for b in range(batch_size):

            # Get actual proposals for this batch element
            n_proposals = num_proposals_batch[b].item()
            actual_actions = actions_batch[b, :n_proposals]
            
            # Compute log probabilities and entropy for this batch element
            action_log_probs[b] = gmm_batch[b].log_prob(actual_actions).sum() # TODO: Is the sum operation correct?
            _, entropy[b] = self.gmm_entropy(gmm_batch[b])
        
        return action_log_probs, state_values_batch, entropy
    
    def gmm_entropy(self, gmm_single):
        """
        """
        entropy_mc = gmm_entropy_monte_carlo(gmm_single)
        entropy_legendre = gmm_entropy_legendre(gmm_single)
        return entropy_mc, entropy_legendre

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
    
    def visualize_gmm(self, gmm_single, num_samples=50000, markers=None, batch_index=None, thickness_range=None, location_range=None, iteration=None):
        """
        Visualize the GMM distribution in 3D.
        If locations are provided, they are marked as red crosses in a separate top-down view.

        Args:
            gmm_single (MixtureSameFamily): The GMM distribution for a single batch element.
            num_samples (int): Number of samples to generate for visualization.
            markers (tuple of ndarrays): Markers to plot, shape (N, 2).
        """
        fs = 16
        base_save_path = f"gmm_iterations/gmm_distribution_iter_{iteration}_batch_{batch_index}"
        
        # Sample from the GMM
        samples = gmm_single.sample((num_samples,))  # Shape: (num_samples, 2)
        samples = samples.detach().cpu().numpy()

        xmin, xmax = location_range
        ymin, ymax = thickness_range
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

        # Main 3D plot without markers
        fig = plt.figure(figsize=(10, 8), dpi=100)   
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.7)
        ax.set_xlabel('Location', fontweight='bold', fontsize=fs, labelpad=15)
        ax.set_ylabel('Thickness', fontweight='bold', fontsize=fs, labelpad=15)
        ax.set_zlabel('Density', fontweight='bold', fontsize=fs, labelpad=15)
        ax.set_title('GMM Distribution', fontweight='bold', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs-2)
        
        # Set fixed z-axis limits
        ax.set_zlim(0, 0.6)
        ax.set_zticks(np.linspace(0, 0.6, 5))
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{base_save_path}.png")
        plt.close()

        # Create second plot with markers if provided
        if markers is not None:
            fig = plt.figure(figsize=(10, 8), dpi=100)
            ax = plt.gca()
            
            ax.grid(True, linestyle=(0, (5, 8)), alpha=0.9, zorder=11)

            contour = ax.contourf(X, Y, Z, levels=20, cmap=cm.viridis, alpha=0.8, zorder=7)
            cbar = plt.colorbar(contour)
            cbar.set_label('Density', fontweight='bold', fontsize=fs)
            cbar.ax.tick_params(labelsize=fs-2)

            locations, thicknesses = markers
            ax.scatter(locations, thicknesses, c='r', marker='x', s=100, label='Samples Drawn', zorder=10)
            legend = ax.legend(loc='upper right', frameon=True, framealpha=1.0)
            legend.set_zorder(11)  
            
            ax.set_xlabel('Location', fontweight='bold', fontsize=fs)
            ax.set_ylabel('Thickness', fontweight='bold', fontsize=fs)
            ax.set_title('GMM with Samples', fontweight='bold', fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs-2)
            plt.tight_layout()
            plt.savefig(f"{base_save_path}_markers.png")
            plt.close()
