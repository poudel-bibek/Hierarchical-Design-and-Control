import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

######## MLP model ########
class MLPActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        """
        A simple MLP Actor-Critic network 
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
    

######## Transformer model ########
class SharedTransformerActorCritic(nn.Module):
    """
    If planning to look at the attention map, the row represents "attending" token, and the column represents "attended to" token.

    What should be a token?
    - An ideal token structure should allow the attention mechanism to focus on meaningful units of information, facilitating the model in finding relevant patterns and relationships.
    Data: size 38
        - 2: traffic phase info
        - 3 x 4: vehicle queue info (incoming)
        - 3 x 4: vehicle queue info (inside)
        - 4: outgoing
        - 4: pedestrian incoming
        - 4: pedestrian outgoing

    - Each token consisting of 3 features, with a single padding of 0 on left for the first token. Making total 39 (divisible by 13) as well as preserving the meaning structure of 3 features for latter tokens.
    - This makes the sequence length 10 x 13 = 130. The attention matrix is 130 by 130
    """

    def __init__(self, state_dim, action_dim, device, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        """
        d_model: Dimensionality of the feature vector for each token. During tokenization, each token is projected into a vector of this size. The output is of tokenization is of shape (batch_size, sequence_length, d_model).
            Hyper-param. Too high = more computation, too low = less capacity to learn complex patterns.
        
        nhead: The number of heads in the multi-head attention models.
            More heads allow the model to jointly attend to information from different representation subspaces.

        num_layers: The number of sub-encoder-layers in the transformer encoder.
                    More layers generally increase the model's capacity but also its computational cost.

        dim_feedforward: In each encoder layer, there's a feedforward network after the multi-head attention mechanism. Typical value (often set to 2-4 times the size of d_model)
                       
        """
        super(SharedTransformerActorCritic, self).__init__()
        self.device = device
        
        # state_dim is 380 (10 timesteps * 38 features per timestep)
        self.state_dim = state_dim

        # Each element in the sequence can attend to all other elements. 
        # Attention mechanism's time complexity is quadratic with respect to the sequence length.
        # For a sequence of length L with each element of dimension d, calculating dot product between all pairs of queries and keys requires (L^2) multiplications and additions. 
        # With a time complexity of O(L^2 x d).
        self.num_timesteps = 10
        self.feature_dim = state_dim // self.num_timesteps  # 38
        self.features_per_token = 3

        # Calculate number of tokens per timestep (including padding)
        self.tokens_per_timestep = math.ceil((self.feature_dim + 1) / self.features_per_token)  # +1 for padding
        self.sequence_length = self.num_timesteps * self.tokens_per_timestep

        # TrafficFlowTokenizer
        self.tokenizer = TrafficFlowTokenizer(state_dim, self.num_timesteps, self.feature_dim, self.features_per_token, d_model)
        
        # Positional encoding: Adds information about token position in sequence
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.sequence_length)
        
        # Shared Transformer layers
        # Create a single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Calculate the size of the flattened output
        self.flattened_size = self.sequence_length * d_model

        # Shared representation layer: Further processes transformer output
        self.shared_layer = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512) # Adding a layer normalization layer for better training stability
        )

        # Actor head: Sequential layers for action logits
        self.actor = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Critic head: Sequential layers for state value estimate 
        self.critic = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    
    def forward(self, state):
        
        # Tokenize and embed the state
        x = self.tokenizer(state)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # After passing through the transformer, the shape is (batch_size, sequence_length, d_model)
        # The slicing operation x = x[:, -1, :] only selects the last token from the sequence. For each sample in a batch, we get last token (which is of size d_model) 
        # Before: x has shape (batch_size, sequence_length, d_model) After: x has shape (batch_size, d_model)
        # In some sequence prediction tasks, the last token is assumed to contain the most relevant information for the final prediction.
        # But in this traffic scenario, information from all time steps is relevant. We want the model to look at the entire sequence.
        # There are potentially other options such as taking the mean of all tokens, or using a pooling operation to aggregate information across all tokens.
        # But we will use flatten. The disadvantage of flattening is that it has higher param count.
        x = x.reshape(x.size(0), -1)  # Shape: (batch_size, sequence_length * d_model)
        # The output of a transformer encoder is also a sequence, flatteing this does not make use of sequence information. But we many not need sequence information further.
        # There are several other considerations such as potential overfitting. Not allowing variable length sequences.
        # The choice depends on specific nature of data, even in this case, the last token of a sequence means information about just one direction/ pedestrian crosswalk. So we use all.
        print(f"\nEncoder output shape: {x.shape}\n")

        # Apply shared representation layer
        shared_features = self.shared_layer(x)

        # Compute actor (action logits) and critic (state value) outputs
        actor_output = self.actor(shared_features)
        critic_output = self.critic(shared_features)
        
        return actor_output, critic_output
    
    def act(self, state):
        # Add batch dimension if not present
        state = state.unsqueeze(0) if state.dim() == 1 else state
        # Get actor output (action logits)
        actor_output, _ = self.forward(state)
        # Create a categorical distribution from logits
        dist = Categorical(logits=actor_output)
        # Sample an action from the distribution
        action = dist.sample()
        # Return the action and its log probability
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, states, actions):
        # Get both actor and critic outputs
        actor_output, critic_output = self.forward(states)
        # Create a categorical distribution from actor logits
        dist = Categorical(logits=actor_output)
        
        # Compute log probabilities of the given actions
        action_logprobs = dist.log_prob(actions)
        # Compute entropy of the action distribution
        dist_entropy = dist.entropy()
        
        return action_logprobs, critic_output, dist_entropy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) for positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Register the positional encoding as a buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class TrafficFlowTokenizer(nn.Module):
    """
    
    """

    def __init__(self, state_dim, num_timesteps, features_per_timestep, features_per_token, d_model):
        super(TrafficFlowTokenizer, self).__init__()
        self.state_dim = state_dim
        self.num_timesteps = num_timesteps
        self.features_per_timestep = features_per_timestep
        self.features_per_token = features_per_token
        self.d_model = d_model
        
        # Calculate number of tokens per timestep (including padding)
        self.tokens_per_timestep = math.ceil((features_per_timestep + 1) / features_per_token)
        
        # Create embeddings for each feature
        self.feature_embeddings = nn.Parameter(torch.randn(features_per_token, d_model))
        
    def forward(self, x):
        # x shape: (batch_size, state_dim)
        batch_size = x.shape[0]
        
        # Reshape to (batch_size, num_timesteps, features_per_timestep)
        x = x.view(batch_size, self.num_timesteps, self.features_per_timestep)
        
        # Pad and reshape to group features into tokens
        padded_x = torch.zeros(batch_size, self.num_timesteps, self.tokens_per_timestep * self.features_per_token, device=x.device)
        padded_x[:, :, 1:self.features_per_timestep+1] = x  # Padding at the leftmost position
        
        grouped_x = padded_x.view(batch_size, self.num_timesteps * self.tokens_per_timestep, self.features_per_token)
        
        # Apply feature embeddings
        embedded = torch.matmul(grouped_x, self.feature_embeddings)
        
        return embedded

# Example usage
# if __name__ == "__main__":
#     state_dim = 380
#     action_dim = 4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = SharedTransformerActorCritic(state_dim, action_dim, device)
    
#     # Count model parameters
#     num_params = sum(p.numel() for p in model.parameters())
#     print(f"Number of parameters: {num_params}")

#     # Simulate input
#     batch_size = 32
#     input_state = torch.rand(batch_size, state_dim)

#     # Forward pass
#     actor_output, critic_output = model(input_state)

#     print(f"Input shape: {input_state.shape}")
#     print(f"Actor output shape: {actor_output.shape}")
#     print(f"Critic output shape: {critic_output.shape}")
#     print(f"Number of tokens per timestep: {model.tokens_per_timestep}")
#     print(f"Total number of tokens in sequence: {model.sequence_length}")