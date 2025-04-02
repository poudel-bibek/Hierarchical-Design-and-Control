import torch
import numpy as np
import torch.nn as nn
from matplotlib import cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical, Bernoulli
from .ppo_utils import gmm_entropy_monte_carlo, gmm_entropy_legendre
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

    def act(self, state, num_proposals):
        """
        Sample an action exactly like in the CNN version:
          - intersection action from first 4 logits (Categorical)
          - midblock from next num_proposals logits (Bernoulli)
          - Ignore the rest of the logits.

        TODO: Is there a bias in log_prob because of the number of proposals?
        How to propoerly handle the rest (is ignoring them good?)
        """
        # print(f"Sampling actions for intersection and {num_proposals} midblock proposals...")
        state = state.reshape(1, 1, state.shape[0], state.shape[1])
        action_logits = self.actor(state)

        # The first 4 logits => intersection (Categorical)
        intersection_logits = action_logits[:, :4]
        # intersection_probs = torch.softmax(intersection_logits, dim=1)
        intersection_dist = Categorical(logits=intersection_logits)
        intersection_action = intersection_dist.sample()  # [1]

        # The next num_proposals logits => midblock (Bernoulli)
        midblock_logits = action_logits[:, 4: 4 + num_proposals]
        # midblock_probs = torch.sigmoid(midblock_logits)
        midblock_dist = Bernoulli(logits=midblock_logits)
        midblock_actions = midblock_dist.sample()  # shape [1,num_proposals]

        # print(f"\nIntersection logits: {intersection_logits}")
        # print(f"\nMidblock logits: {midblock_logits}")

        # print(f"\nIntersection action: {intersection_action}")
        # print(f"\nMidblock actions: {midblock_actions}")
        
        combined_action = torch.cat([intersection_action, midblock_actions.squeeze(0)], dim=0)
        # print(f"\nCombined action: {combined_action}")

        log_prob = intersection_dist.log_prob(intersection_action) + \
                   midblock_dist.log_prob(midblock_actions).sum()

        # print(f"\nAction Log probability: {log_prob}, shape: {log_prob.shape}")
        return combined_action.int(), log_prob


    def evaluate(self, states, actions, num_proposals, device=None):
        """
        Evaluate a batch of states and pre-sampled actions. 
        number of proposals remains same for all parallel actors that collect experiences. 
        
        TODO: Remove the bias due to number of proposals.
        """
        print("Evaluating... with num_proposals: ", num_proposals)
        action_logits = self.actor(states)
        intersection_logits = action_logits[:, :4]
        midblock_logits = action_logits[:, 4: 4 + num_proposals]

        # Distributions
        # intersection_probs = torch.softmax(intersection_logits, dim=1)
        intersection_dist = Categorical(logits=intersection_logits)
        # midblock_probs = torch.sigmoid(midblock_logits)
        midblock_dist = Bernoulli(logits=midblock_logits)
        print(f"\nMidblock dist: {midblock_dist}\n")

        # Actions in shape (B,1) for intersection, (B,num_proposals) for midblock
        intersection_action = actions[:, :1].squeeze(1).long() # Categorical expects long
        midblock_actions = actions[:, 1: 1 + num_proposals].float()

        print(f"\nMidblock logits: {midblock_logits}, midblock actions: {midblock_actions.shape}\n")

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
    GATv2 (with edge features).
    Graph attention layers are shared between actor and critic.

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

    def __init__(self, in_channels, action_dim, **kwargs):
        """
        in_channels: Number of input features per node (e.g., x and y coordinates)
        hidden_channels: Number of hidden features.
        out_channels: Number of output features.
        initial_heads: Number of attention heads for the first GAT layer.
        second_heads: Number of attention heads for the second GAT layer.
        edge_dim: Number of features per edge
        action_dim is the max number of proposals. 

        TODO: 
        # At every timestep, the actions is a whole bunch of things of max size. Critic has to evaluate all that (insted of just the relevant parts).
        # Thickness and location values that are not in the proposal are set to -1 (which is close to minimum of 0.1)
        # model could potentially interpret these as meaningful values. 

        """
        super(GAT_v2_ActorCritic, self).__init__()
        self.in_channels = in_channels
        self.max_proposals = action_dim
        
        self.num_mixtures = kwargs.get('num_mixtures')
        self.hidden_channels = kwargs.get('hidden_channels')
        self.out_channels = kwargs.get('out_channels')
        self.initial_heads = kwargs.get('initial_heads')
        self.second_heads = kwargs.get('second_heads')
        self.edge_dim = kwargs.get('edge_dim')

        self.dropout_rate = 0.0 # kwargs.get('dropout_rate', 0.2)
    
        if kwargs.get('activation') == "elu":
            self.activation = nn.ELU()
        elif kwargs.get('activation') == "relu":
            self.activation = nn.ReLU()
        elif kwargs.get('activation') == "tanh":
            self.activation = nn.Tanh()
        elif kwargs.get('activation') == "leakyrelu":
            self.activation = nn.LeakyReLU()

        model_size = kwargs.get('model_size')
        # Actor has some shared MLP layers
        # Followed by four heads: 
        #     1. Predict GMM means (2 values: location and thickness)
        #     2. Predict log-std (2 values: log-std in location and thickness)
        #     3. Predict mix logits (1 value: weight of each Gaussian component)
        #     4. Predict the number of times to sample from the GMM.
        if model_size == "small":
            actor_shared_hidden_sizes = [256, 128] 
            actor_gmm_hidden_sizes = [128, 64] # For actor heads 1, 2, 3
            actor_sample_hidden_sizes = [128, 64] # For actor head 4
            critic_hidden_sizes = [128, 64]
        elif model_size == "medium":
            actor_shared_hidden_sizes = [512, 256]
            actor_gmm_hidden_sizes = [256, 128, 64]
            actor_sample_hidden_sizes = [256, 128, 64]
            critic_hidden_sizes = [256, 128, 64]

        # First Graph Attention Layer. 
        # conv1 should output [num_nodes, hidden_channels * initial_heads]
        # concat=True by default: the outputs from different attn heads are concatenated to output of size hidden_channels * initial_heads.
        # When concat=False, the outputs from different heads are averaged to output of size: hidden_channels. Reduces the dimensionality of the output
        self.conv1 = GATv2Conv(self.in_channels, 
                               self.hidden_channels, 
                               edge_dim = self.edge_dim, 
                               heads = self.initial_heads, 
                               concat = True, 
                               dropout = self.dropout_rate) 

        # Second Graph Attention Layer 
        # Why heads=1? Often, multi-head attention is used in earlier layers to capture different aspects of the graph and the final layer consolidates this info.
        # conv2 should output [num_nodes, out_channels * second_heads] (when concat = True)
        # conv2 should output [num_nodes, out_channels] (when concat = False). This loses too much information.
        self.conv2 = GATv2Conv(self.hidden_channels * self.initial_heads, 
                               self.out_channels, 
                               edge_dim = self.edge_dim, 
                               heads = self.second_heads, 
                               concat = True, 
                               dropout = self.dropout_rate)

        # actor
        # shared
        actor_shared_layers = []
        input_size_actor_shared = self.out_channels * self.second_heads
        for h in actor_shared_hidden_sizes:
            actor_shared_layers.append(layer_init(nn.Linear(input_size_actor_shared, h)))
            # Add layer norm, batch norm, dropout, etc.
            actor_shared_layers.append(nn.LayerNorm(h))
            actor_shared_layers.append(self.activation)
            # actor_shared_layers.append(nn.Dropout(self.dropout_rate))
            input_size_actor_shared = h

        self.actor_shared_layers = nn.Sequential(*actor_shared_layers)

        # means
        actor_means_layers = []
        input_size_actor_means = input_size_actor_shared
        for h in actor_gmm_hidden_sizes:
            actor_means_layers.append(layer_init(nn.Linear(input_size_actor_means, h)))
            # Add layer norm, batch norm, dropout, etc.
            actor_means_layers.append(nn.LayerNorm(h))
            actor_means_layers.append(self.activation)
            # actor_means_layers.append(nn.Dropout(self.dropout_rate))
            input_size_actor_means = h

        self.actor_means_layers = nn.Sequential(*actor_means_layers)
        self.actor_means = layer_init(nn.Linear(input_size_actor_means, self.num_mixtures * 2))

        # log-std
        actor_log_std_layers = []
        input_size_actor_log_std = input_size_actor_shared
        for h in actor_gmm_hidden_sizes:
            actor_log_std_layers.append(layer_init(nn.Linear(input_size_actor_log_std, h)))
            # Add layer norm, batch norm, dropout, etc.
            actor_log_std_layers.append(nn.LayerNorm(h))
            actor_log_std_layers.append(self.activation)
            # actor_log_std_layers.append(nn.Dropout(self.dropout_rate))
            input_size_actor_log_std = h

        self.actor_log_std_layers = nn.Sequential(*actor_log_std_layers)
        self.actor_log_std = layer_init(nn.Linear(input_size_actor_log_std, self.num_mixtures * 2))

        # mix logits
        actor_mix_logits_layers = []
        input_size_actor_mix_logits = input_size_actor_shared
        for h in actor_gmm_hidden_sizes:
            actor_mix_logits_layers.append(layer_init(nn.Linear(input_size_actor_mix_logits, h)))
            # Add layer norm, batch norm, dropout, etc.
            actor_mix_logits_layers.append(nn.LayerNorm(h))
            actor_mix_logits_layers.append(self.activation)
            # actor_mix_logits_layers.append(nn.Dropout(self.dropout_rate))
            input_size_actor_mix_logits = h

        self.actor_mix_logits_layers = nn.Sequential(*actor_mix_logits_layers)
        self.actor_mix_logits = layer_init(nn.Linear(input_size_actor_mix_logits, self.num_mixtures))

        # number of proposals (Does not share i.e., attached directly to readout layer output)
        actor_sample_layers = []
        input_size_actor_sample = self.out_channels * self.second_heads
        for h in actor_sample_hidden_sizes:
            actor_sample_layers.append(layer_init(nn.Linear(input_size_actor_sample, h)))
            # Add layer norm, batch norm, dropout, etc.
            actor_sample_layers.append(nn.LayerNorm(h))
            actor_sample_layers.append(self.activation)
            # actor_gmm_layers.append(nn.Dropout(self.dropout_rate))
            input_size_actor_sample = h

        self.actor_sample_layers = nn.Sequential(*actor_sample_layers)
        self.actor_num_proposals = layer_init(nn.Linear(input_size_actor_sample, self.max_proposals)) # No activation in last layer.

        # critic
        critic_layers = []
        input_size_critic = self.out_channels * self.second_heads
        for h in critic_hidden_sizes:
            critic_layers.append(layer_init(nn.Linear(input_size_critic, h)))
            # Add layer norm, batch norm, dropout, etc.
            critic_layers.append(nn.LayerNorm(h))
            critic_layers.append(self.activation)
            # critic_layers.append(nn.Dropout(self.dropout_rate))
            input_size_critic = h

        self.critic_layers = nn.Sequential(*critic_layers)
        self.critic_value = layer_init(nn.Linear(input_size_critic, 1))

    def readout_layer(self, x, batch):
        """
        Applied at the transition of GAT to MLP layers.
        Without it, the expected shape is (num_nodes * out_channels * second_heads): num_nodes can be different for each graph.
        A number of approaches are possible.
        - global_mean_pool: average across the nodes for each graph in a batch.

        """
        return global_mean_pool(x, batch)

    def actor(self, states_batch):
        """
        - GMM parameters prediction 
        - Number of proposals prediction (# of times to sample from GMM)

        State = Data or Batch object with 4 tensors:
        - node features (x) = (num_nodes, in_channels)
        - edge index (edge_index) = (2, num_edges) connections between nodes. # e.g., edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
        - edge attributes (edge_attr) = Edge features (num_edges, edge_dim)
        - batch (batch)
        """

        y = self.conv1(states_batch.x, states_batch.edge_index, states_batch.edge_attr)
        y = self.activation(y)

        # The same edge_attr is used for both the first and second GAT layers. 
        # Preserving edge information: By passing the edge attributes to each layer, the model can maintain information about the edge features throughout the network depth. 
        # Different learned attention: Each GAT layer learns its own attention mechanism. By providing the edge features to each layer, you allow each layer to learn how to use these features differently
        # Residual-like connections for edges: In a sense, passing edge features to each layer creates a form of residual connection for edge information.         
        y = self.conv2(y, states_batch.edge_index, states_batch.edge_attr)
        y = self.activation(y)

        y = self.readout_layer(y, states_batch.batch)

        # Shared MLP layers
        shared_y = self.actor_shared_layers(y)

        # GMM parameters. Constrain so that the GMM is within the boundary.
        means = torch.sigmoid(self.actor_means(self.actor_means_layers(shared_y)))  # Constrain means to [0,1]
        
        # Scale and shift tanh to get log_stds in [-2.30, 0] range
        # tanh outputs [-1, 1], so we scale by 0.693 and shift by -2.303 to get [-3.0, -0.693], exp(-3.0) ≈ 0.05, exp(-0.693) ≈ 0.5
        log_stds = torch.tanh(self.actor_log_std(self.actor_log_std_layers(shared_y))) * 0.693 - 2.303  # Results in std range [0.05, 0.5]

        # # tanh outputs [-1, 1], so we scale by 1.15 and shift by -1.15 to get [-2.30, 0], exp(-2.30) ≈ 0.1, exp(0) = 1.
        # log_stds = torch.tanh(self.actor_log_std(self.actor_log_std_layers(shared_y))) * 1.15 - 1.15  # Results in std range [0.1, 1.0]

        mix_logits = self.actor_mix_logits(self.actor_mix_logits_layers(shared_y))
        num_proposal_logits = self.actor_num_proposals(self.actor_sample_layers(y)) # probabilites obtained later.

        return means, log_stds, mix_logits, num_proposal_logits

    def critic(self, states_batch):
        """
        """
        y = self.conv1(states_batch.x, states_batch.edge_index, states_batch.edge_attr)
        y = self.activation(y)
        y = self.conv2(y, states_batch.edge_index, states_batch.edge_attr)
        y = self.activation(y)

        y = self.readout_layer(y, states_batch.batch)   
        critic_features = self.critic_layers(y)
        return self.critic_value(critic_features).squeeze(-1)  # Ensure output is of shape (batch_size,)

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

        # In torch geometric, a batch of graphs is a single large graph. Different from gradient mini-batch.
        # batch_size = 1 looks like [0, 0, 0, 0, ... num_nodes times]
        # batch_size = 4 looks like [0, 0, 1, 1, 2, 2, 3, 3, ... num_nodes times]
        batch_size = states_batch.num_graphs 
        # print(f"\nInside get_gmm_distribution:\nBatch size: {batch_size}")

        # Returns (batch_size, num_mixtures * 5) and (batch_size, max_proposals)
        means, log_stds, mix_logits, num_proposals_logits = self.actor(states_batch)
        # print(f"\nMeans: {means}, Log-stds: {log_stds}, Mix logits: {mix_logits}, Num proposal logits: {num_proposals_logits}")

        num_proposals_probs_batch = F.softmax(num_proposals_logits, dim=-1)
        # print(f"\nProposal probabilities: {num_proposals_probs_batch}")

        gmms_batch = []
        for b in range(batch_size):
            # reshape to (num_mixtures, 2)
            means_b = means[b].reshape(self.num_mixtures, 2)
            log_stds_b = log_stds[b].reshape(self.num_mixtures, 2)
            # print(f"\nMeans item {b}: {means_b} \nLog-stds item {b}: {log_stds_b}")
            
            # diagonal covariance matrix
            scale_diag = torch.exp(log_stds_b) # exponentiate the logstdss
            mixture_dist = Categorical(logits = mix_logits[b])

            component_dist = MultivariateNormal(
                loc = means_b, 
                scale_tril = torch.diag_embed(scale_diag) 
            )
            gmm = MixtureSameFamily(mixture_distribution = mixture_dist,
                                    component_distribution = component_dist)
            gmms_batch.append(gmm)
        return gmms_batch, num_proposals_probs_batch
    
    def _sample_gmm(self, gmm_single, num_proposals, naive_stochastic = False, training = True, device = None):
        """

        Sampling from GMM has several issues of discussion.
        1. During training: Stochastic vs Deterministic sampling:
        * Make use of the implicit stochasticity (set naive_stochastic = True)
          Justification: PG methods rely heavily on action space exploration (to get diverse experiences) within the distribution defined by the policy parameters.
          Algorithms like PPO are foundationally built around optimizing a stochastic policy.

        * Greedily sample at highest probability (modes), deterministic action selection.
          Justification: The policy can do exploration in its parameter space. The sampling does not need to introduce additional stochasticity.

            - Discretize: Divide the GMM into 10 x 10 grid.
            - Evaluate: Calculate the probability at each grid cell
            - Rank and select: Find the top num_proposals number of cells with highest probability.
            - Sample: Use the center of the cell to get a sample.

            The problem here is that: the selected cells may always be near the model with highest probability.
            When the GMM is multi-modal, we would instead want the selected cells to be spread out across multiple modes.

        2. Sample stochastically at train vs deterministically at test time?
        * Making the design agent stochastic at test time does make sense for this problem.
          Justification: The design actions are more like "one-shot" generation unlike the control actions (which are continouus and episodic). 
          The design policy finishes training and we are supposed to get the best design at the end.
          It is kind of doing one round of expectation minimization.

        """
        if training:
            if naive_stochastic:
                return gmm_single.sample((num_proposals,))
            
            else:
                pass

        else: # Test time
            pass

    def act(self, states_batch, iteration, clamp_min, clamp_max, device, training = True, visualize=False):
        """
        Sample actions from the GMM (propose upto max_proposals number of crosswalks).
        Policy gradient methods require the log probabilities of the actions to be returned as well.
        Should thickness and location be independent? 
        - No. 
        - Modeling thickness and location jointly 
        Utilizing the implicit stochasticity in sampling during training.
        However, at test, sample greedily at each mode.

        TODO: Whether to make the policy deterministic at test time?
        """

        # properly batch the data using Batch.from_data_list() before sending here. 
        gmm_batch, num_proposals_probs_batch = self.get_gmm_distribution(states_batch.to(device))
        # print(f"\n\nGMMs: {gmm_batch}\n\n")

        # Sample one proposal for each batch element (add 1 to ensure at least 1 proposal; default starts from index 0)
        # Using torch multinomial to sample from discrete distribution.
        num_proposals = torch.multinomial(num_proposals_probs_batch, 1).squeeze(-1) + 1
        # print(f"\nnum_proposals: {num_proposals.shape, num_proposals}\n")

        batch_size = states_batch.num_graphs 
        # Initialize output tensors with -1 so that its easier to infer actual proposals in critic.
        padded_proposals = torch.full((batch_size, self.max_proposals, 2), -1.0, dtype=torch.float32, device=device) 
        log_probs = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):
            # Sample proposals for this batch element
            samples = self._sample_gmm(gmm_batch[b], num_proposals[b].item(), naive_stochastic = True, training = training, device = device)

            locations, thicknesses = samples.split(1, dim=-1)
            # print(f"\nBefore clamping: Locations: {locations}, Thicknesses: {thicknesses}")
        
            # Apply a noisy clamp individually to prevent exact overlap at same locations.
            # Although means are constrained to [0,1], log_stds are not.
            # We should be less dependent on clamping here and make sure the GMM itself lies in the desired range.
            below_min_mask_loc = locations < clamp_min
            above_max_mask_loc = locations > clamp_max
            locations[below_min_mask_loc] = 0.0 + torch.rand_like(locations[below_min_mask_loc], device=device) * clamp_min # In the range [0, clamp_min]
            locations[above_max_mask_loc] = clamp_max + torch.rand_like(locations[above_max_mask_loc], device=device) * (1.0 - clamp_max) # In the range [clamp_max, clamp_max + noise_offset]
            
            # Standard clamp
            # locations = torch.clamp(locations, clamp_min, clamp_max)  
            
            thicknesses = torch.clamp(thicknesses, clamp_min, clamp_max) 

            # print(f"\nAfter clamping: Locations: {locations}, Thicknesses: {thicknesses}")
 
            # Recombine the samples
            samples = torch.cat([locations, thicknesses], dim=-1)
            
            # Visualization is only meaningful during act (i.e., not during evaluation)
            if visualize and iteration is not None:
                markers = (locations.squeeze().detach().cpu().numpy(), thicknesses.squeeze().detach().cpu().numpy())
                self.visualize_gmm(gmm_batch[b], markers=markers, batch_index=b, thickness_range=(0, 1), location_range=(0, 1), iteration=iteration)

            # Store in output tensor
            padded_proposals[b, :num_proposals[b], 0] = locations.squeeze()
            padded_proposals[b, :num_proposals[b], 1] = thicknesses.squeeze()
            
            # Compute log probabilities for this batch element
            # SUM operation is correct for joint log prob of thickness and location.
            log_probs[b] = gmm_batch[b].log_prob(samples).sum() 

        return padded_proposals, num_proposals, log_probs
    
    def evaluate(self, states_batch, actions_batch, device = None):
        """
        states_batch (Batch): Batch of states, each state a Data object.
        actions_batch (Tensor): Batch of actions [batch_size, max_proposals, 2]. 
            - Not all the actions contain actual proposals. Each element in the batch can have a different number of proposals.

        Returns:
            action_log_probs (Tensor): Log probabilities of the actions.
            state_values (Tensor): Values of the states.
            entropy (Tensor)
        """
    
        state_values = self.critic(states_batch.to(device))
        # Get distribution (internally passes through actor)
        gmm_batch, _ = self.get_gmm_distribution(states_batch.to(device))
        
        # Initialize return tensors
        batch_size = states_batch.num_graphs 
        action_log_probs = torch.zeros(batch_size, device=device)
        entropy = torch.zeros(batch_size, device=device)
        
        # print(f"\nActions batch: {actions_batch}\n")
        # Compute num_proposals_batch by checking for -1 in actions
        num_proposals_batch = (actions_batch[:, :, :, 0] != -1).sum(dim=2) # squeeze reduces to two dimensions.
        # print(f"\nNum proposals batch: {num_proposals_batch}\n")

        for b in range(batch_size):
            # Get actual proposals for this batch element
            n_proposals = num_proposals_batch[b].item()
            actual_actions = actions_batch[b, :n_proposals]
            
            # Compute log probabilities and entropy for this batch element
            # The sum operation ensures joint log prob of thickness and location.
            action_log_probs[b] = gmm_batch[b].log_prob(actual_actions).sum()
            _, entropy[b] = self.get_entropy(gmm_batch[b])
        
        return action_log_probs, state_values, entropy
    
    def get_entropy(self, gmm_single):
        """
        """
        entropy_mc = gmm_entropy_monte_carlo(gmm_single)
        # entropy_legendre = gmm_entropy_legendre(gmm_single)
        return None, entropy_mc

    def param_count(self):
        """
        Count the total number of parameters in the model.
        """
        # Shared GATv2 layers
        shared_gat_params = sum(p.numel() for p in self.conv1.parameters()) + \
                        sum(p.numel() for p in self.conv2.parameters())

        # Shared MLP layers in actor
        shared_mlp_params = sum(p.numel() for p in self.actor_shared_layers.parameters())

        # Actor-specific 
        actor_means_params = sum(p.numel() for p in self.actor_means_layers.parameters())
        actor_log_std_params = sum(p.numel() for p in self.actor_log_std_layers.parameters())
        actor_mix_logits_params = sum(p.numel() for p in self.actor_mix_logits_layers.parameters())
        actor_sample_params = sum(p.numel() for p in self.actor_sample_layers.parameters())

        actor_params = actor_means_params + actor_log_std_params + actor_mix_logits_params + actor_sample_params

        # Critic-specific 
        critic_params = sum(p.numel() for p in self.critic_layers.parameters())

        total_params = shared_gat_params + shared_mlp_params + actor_params + critic_params 
        return {
            "shared_gat": shared_gat_params,
            "actor_shared_mlp": shared_mlp_params,
            "actor_means": actor_means_params,
            "actor_log_std": actor_log_std_params,
            "actor_mix_logits": actor_mix_logits_params,
            "actor_sample": actor_sample_params,
            "actor_total": actor_params,
            "critic": critic_params,
            "Grand total": total_params}
    
    def visualize_gmm(self, gmm_single, num_samples=50000, markers=None, batch_index=None, thickness_range=None, location_range=None, iteration=None):
        """
        Visualize the GMM distribution in 3D.
        If sampling locations provided, they are marked as red crosses in a separate top-down view.

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
        ax.set_zlim(0, 1.0)
        ax.set_zticks(np.linspace(0, 1.0, 5))
        
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

            # Get and plot component means
            means = gmm_single.component_distribution.loc.detach().cpu().numpy()
            ax.scatter(means[:, 0], means[:, 1], c='blue', marker='o', s=120, edgecolors='black', label='Component Means', zorder=11)

            locations, thicknesses = markers
            ax.scatter(locations, thicknesses, c='r', marker='x', s=100, label='Samples Drawn', zorder=10)
            legend = ax.legend(loc='upper right', frameon=True, framealpha=1.0)
            legend.set_zorder(12) # Ensure legend is on top
            
            ax.set_xlabel('Location', fontweight='bold', fontsize=fs)
            ax.set_ylabel('Thickness', fontweight='bold', fontsize=fs)
            ax.set_title('GMM with Samples', fontweight='bold', fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs-2)
            plt.tight_layout()
            plt.savefig(f"{base_save_path}_markers.png")
            plt.close()
