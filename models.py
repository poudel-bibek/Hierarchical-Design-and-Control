import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Bernoulli
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.data import Data
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
        receive all other than in_channels and action_dim as kwargs
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

        # Store initialization parameters as instance variables
        self.max_proposals = action_dim
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.num_mixtures = num_mixtures
        self.dropout_rate = dropout_rate

        if hidden_channels is None:
            print(f"\nGAT initial hidden channels not specified !!\n")
        print(f"in channels: {in_channels}")
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

        self.elu = nn.ELU()
        
    def readout_layer(self, x, batch):
        """
        As a number of approaches are possible, this is a separate function.
        """
        # global_mean_pool to average across the nodes for each graph in a batch.
        return global_mean_pool(x, batch)


    def actor(self, state):
        """
        This is the Actor forward pass, which consists of two parts (all in one head):
        - GMM parameters prediction layers
        - Number of proposals prediction layer (how many times to sample from the GMM)

        Instead of accepting Data object, accepts 4 tensors:
        - node features (x) = (num_nodes, in_channels)
        - edge index (edge_index) = (2, num_edges) connections between nodes. # e.g., edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
        - edge attributes (edge_attr) = Edge features (num_edges, edge_dim)
        - batch (batch) = 
        """
        
        print(f"Data x device: {state.x.device}")
        print(f"Model device: {next(self.parameters()).device}")

        y = self.elu(self.conv1(state.x, state.edge_index, state.edge_attr))

        # Apply dropout to hidden representations
        y = F.dropout(y, p=self.dropout_rate, training=self.training)

        # The same edge_attr is used for both the first and second GAT layers. 
        # Apply second GAT layer with edge features (no activation here as it's the final layer)
        # Why are edge features passed again?
        # Preserving edge information: By passing the edge attributes to each layer, the model can maintain information about the edge features throughout the network depth. 
        # Different learned attention: Each GAT layer learns its own attention mechanism. By providing the edge features to each layer, you allow each layer to learn how to use these features differently
        # Residual-like connections for edges: In a sense, passing edge features to each layer creates a form of residual connection for edge information.         
        y = self.conv2(y, state.edge_index, state.edge_attr)

        y = self.readout_layer(y, state.batch)

        gmm_params = self.actor_gmm_layers(y)
        num_proposals_logits = self.actor_num_proposals_layer(y)

        return gmm_params, num_proposals_logits

    def critic(self, state):
        """
        This is the Critic forward pass.
        """
        x, edge_index, edge_attr, batch = state

        y = nn.elu(self.conv1(x, edge_index, edge_attr))
        y = F.dropout(y, p=self.dropout_rate, training=self.training)
        y = self.conv2(y, edge_index, edge_attr)
        y = self.readout_layer(y, batch)

        return self.critic_layers(y)
    
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
        samples = gmm.sample((num_samples,))
        
        # Compute log probabilities
        log_probs = gmm.log_prob(samples)
        entropy = -log_probs.mean() # Approximate entropy
        return entropy

    def visualize_gmm(self, gmm, num_samples=20000, save_path=None):
        """
        Visualize the GMM distribution using Seaborn and Matplotlib.
        """
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Sample from the GMM
        samples = gmm.sample((num_samples,))
        print(f"\nsamples: {samples.shape}\n")

        # Ensure samples are 2D
        if samples.dim() == 3:
            samples = samples.squeeze(1)

        sns.scatterplot(x=samples[:, 0], y=samples[:, 1], alpha=0.5, ax=ax)
        ax.set_xlabel("Location", fontsize=12)
        ax.set_ylabel("Thickness", fontsize=12)
        ax.set_title("Gaussian Mixture Model", fontsize=14)
        
        # Adjust x-axis
        #ax.set_xlim(-0.5, 1.5)
        #ax.set_xticks(np.arange(-0.5, 1.5, 0.1))
        #ax.set_xticklabels([f"{x:.1f}" for x in np.arange(-0.5, 1.5, 0.1)])
        
        # Adjust y-axis
        #ax.set_ylim(self.min_thickness, self.max_thickness)
        #y_ticks = np.linspace(self.min_thickness, self.max_thickness, 10)
        #ax.set_yticks(y_ticks)
        #ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])
        
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="YlOrRd", fill=True, cbar=True, ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GMM distribution plot saved to {save_path}")
        plt.close()
    
    def get_gmm_distribution(self, state):
        """
        Construct the GMM distribution given the state.
        Also get the actual number of proposals.
        Since this is repeated, it's a function.
        """
        # Get GMM parameters and number of proposals (logits)
        gmm_params, num_proposals_logits = self.actor(state)
        print(f"\nnum_proposals_logits: {num_proposals_logits}\n")

        # Apply temperature to control exploration-exploitation trade-off
        num_proposals_probs = F.softmax(num_proposals_logits / self.temperature, dim=-1)  # Convert to probabilities for each index (total sum to 1) with temperature

        # Construct the GMM distribution
        mix_logits, means, covs = gmm_params.split([self.num_mixtures, self.num_mixtures * 2, self.num_mixtures * 2], dim=-1)
        
        means = means.view(-1, self.num_mixtures, 2)
        covs = F.softplus(covs).view(-1, self.num_mixtures, 2) # Ensure positive covariance

        mix = Categorical(logits=mix_logits) # Categorical distribution for the mixture probabilities
        covariance_matrices = torch.diag_embed(covs) # Create diagonal covariance matrices
        comp = MultivariateNormal(means, covariance_matrices) # Multivariate normal distributions for each component
        gmm = MixtureSameFamily(mix, comp) # Mixture of Gaussians distribution

        return gmm, num_proposals_probs

    def act(self, x, edge_index, edge_attr, batch):
        """
        Propose up to max_proposals number of crosswalks.
        For use in policy gradient methods, the log probabilities of the actions are needed.

        We are using reparameterization trick (which assumes that the the actions follow a certain continuous and differentiable distribution)
        By default its normal distribution. The problem with normal distribution is that it assumes a single mode. When sampling, likelihood of getting a sample far away from the mean is low (depends on std).
        Instead, we use a mixture of Gaussians. 
            - Can model more complex distributions
            - Can capture multiple modes in the distribution
            - Flexibility: Can be parameterized to have different means and variances for each component

        Should thickness and location be independent? No. Particular thickness for a specific location is what is needed. 
        Hence, the distribution jointly models the two (location and thickness). 
        """
        if batch is None:  # Assume a single graph is used to make an inference at a time.
            batch = torch.zeros(x.size(0), dtype=torch.long).to(x.device)

        # Stored as tuples with zeroth item being the actual data item.
        gmm, num_proposals_probs = self.get_gmm_distribution(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)) # Internally passes through the actor.

        # multinomial distribution is used to model the outcome of selecting one option from a set of mutually exclusive options, where each option has a specific probability of being chosen.
        # Sample the number of proposals (add 1 to ensure at least 1 proposal)
        num_actual_proposals = torch.multinomial(num_proposals_probs, 1).item() + 1 # The inner 1 to the number of draws
        print(f"\nnum_actual_proposals: {num_actual_proposals}\n")

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

    def evaluate(self, state, action):
        """
        - Evaluate the state (x, edge_index, edge_attr, batch)
        - Return action (max_proposals, 2) log probabilities, state values, and entropy.
        """

        gmm, num_proposals_probs, returned_state = self.get_gmm_distribution(state)

        # Compute log probabilities of the actions
        action_log_probs = gmm.log_prob(action)

        # Compute entropy (how uncertain or random the actions selection is)
        entropy = self.gmm_entropy(gmm) # already performs the mean.
        print(f"\nentropy: {entropy}\n")

        # Compute state value
        state_value = self.critic(returned_state) # critic only takes state.
        print(f"\nState Value: {returned_state.shape}\n")
        
        return action_log_probs, state_value, entropy, num_proposals_probs
    
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