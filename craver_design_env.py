import random
import gymnasium as gym
import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from itertools import tee
from craver_config import (PHASES, DIRECTIONS_AND_EDGES, CONTROLLED_CROSSWALKS_DICT, initialize_lanes, get_phase_groups)
import torch
from torch_geometric.data import Data
from models import GATv2ActorCritic  # Import the GAT model

def pairwise(iterable):
    """
    Generates consecutive pairs from an iterable.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class CraverDesignEnv(gym.Env):
    """
    For the higher level agent, modifies the net file based on the design decision.
    No need to connect or close this environment. Will be limited to network file modifications.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.original_net_file = './SUMO_files/original_craver_road.net.xml'

        self.crosswalks_to_remove = [crosswalk_id for _, data in CONTROLLED_CROSSWALKS_DICT.items() for crosswalk_id in data['ids']]
        #self._clear_corridor(self.crosswalks_to_remove)
        
        self.pedestrian_graph = self._extract_original_graph()
        self._visualize_pedestrian_graph()
        
        # Initialize normalizer values
        self.normalizer_x = None
        self.normalizer_y = None
        self.normalizer_width = None
        
        # Convert the NetworkX graph to a PyTorch Geometric Data object
        self.torch_graph = self._convert_to_torch_geometric()

    @property
    def action_space(self):
        return gym.spaces.MultiDiscrete([2] * 15)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(10, 74), dtype=np.float32)

    def _extract_original_graph(self):
        """
        Extracts the pedestrian graph from the original SUMO network file, including walking areas, connections, and junctions.
        """
        G = nx.Graph()
        
        tree = ET.parse(self.original_net_file)
        root = tree.getroot()

        # Extract junctions
        for junction in root.findall(".//junction"):
            junction_id = junction.get('id')
            x, y = float(junction.get('x')), float(junction.get('y'))
            junction_type = junction.get('type')
            
            G.add_node(junction_id, pos=(x, y), type=junction_type, width=0.0)

        # Connect junctions based on edges and extract edge widths
        for edge in root.findall(".//edge"):
            from_junction = edge.get('from')
            to_junction = edge.get('to')
            if from_junction in G.nodes() and to_junction in G.nodes():
                # Extract the width of the first lane (assuming all lanes have the same width)
                lane = edge.find('lane')
                if lane is not None:
                    width = float(lane.get('width', 0.0))
                    G.add_edge(from_junction, to_junction, width=width)
                    
                    # Update the width of the nodes (junctions) connected by this edge
                    G.nodes[from_junction]['width'] = max(G.nodes[from_junction]['width'], width)
                    G.nodes[to_junction]['width'] = max(G.nodes[to_junction]['width'], width)
                else:
                    G.add_edge(from_junction, to_junction, width=0.0)

        return G

    def step(self, action):
        """
        Implement the environment step here.
        You can use self.torch_graph as input to your GAT model.
        """
        # Your step implementation
        pass

    def _apply_action(self, action):
        pass
    
    def _get_reward(self, action):
        pass

    def reset(self):
        """
        Implement the environment reset here.
        You can reinitialize self.torch_graph if needed.
        """
        # Your reset implementation
        return self.torch_graph  # Return the initial state

    def close(self):
        """
        Probably wont make use of it, just for completeness.
        """
        pass
    
    def _clear_corridor(self, crosswalks_to_remove):
        """
        Step 1: We remove all existing (given from the real-world) unsignalized crosswalks from the corridor.
        There may be more crosswalks than we care about, hence we remove them using the list.
        """
        # Use Netconvert to remove the crosswalks.
        print(f"Following crosswalks will be removed: {crosswalks_to_remove}.\n")
        pass
    
    def _add_corridor_crosswalks(self):
        """
        Step 2: Based on the current action representation, add number_of_crosswalks with a thickness_of_crosswalks.
        """
        #number_of_crosswalks, thickness_of_crosswalks  = # some function call/
        pass

    def _visualize_pedestrian_graph(self, save_path='pedestrian_graph.png', show=True):
        """
        Visualizes the pedestrian graph of junctions.
        """
        plt.figure(figsize=(20, 15))
        
        pos = nx.get_node_attributes(self.pedestrian_graph, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.pedestrian_graph, pos, node_size=30, node_color='slateblue', alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.pedestrian_graph, pos, edge_color='orange', width=2, alpha=0.2)
        
        plt.title("Interconnected Junctions Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def _convert_to_torch_geometric(self):
        """
        Converts the NetworkX graph to a PyTorch Geometric Data object.
        Normalizes the coordinates to lie between 0 and 1 and scales the width values proportionally.
        """
        # Extract node features (x, y coordinates, width)
        node_features = []
        for node, data in self.pedestrian_graph.nodes(data=True):
            node_features.append([data['pos'][0], data['pos'][1], data['width']])

        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Normalize features
        x = self._normalize_features(x)
        
        # Extract edge indices and attributes
        edge_index = []
        edge_attr = []
        for edge in self.pedestrian_graph.edges(data=True):
            source = list(self.pedestrian_graph.nodes()).index(edge[0])
            target = list(self.pedestrian_graph.nodes()).index(edge[1])
            edge_index.append([source, target])
            edge_index.append([target, source])  # Add reverse edge for undirected graph
            
            width = edge[2]['width']
            # Normalize edge width
            normalized_width = (width - self.normalizer_width['min']) / (self.normalizer_width['max'] - self.normalizer_width['min'])
            edge_attr.append([normalized_width, 0.0])  # Add a dummy feature (0.0) alongside normalized width
            edge_attr.append([normalized_width, 0.0])  # For the reverse edge
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data

    def _normalize_features(self, features):
        """
        Normalizers are gotten once from the original graph.
        Normalize the features (coordinates and width) to lie between 0 and 1.
        Save the normalizer values for potential later use.
        """
        x_coords = features[:, 0]
        y_coords = features[:, 1]
        widths = features[:, 2]

        # Compute normalizers
        self.normalizer_x = {'min': x_coords.min(), 'max': x_coords.max()}
        self.normalizer_y = {'min': y_coords.min(), 'max': y_coords.max()}
        self.normalizer_width = {'min': widths.min(), 'max': widths.max()}

        # Normalize features
        normalized_x = (x_coords - self.normalizer_x['min']) / (self.normalizer_x['max'] - self.normalizer_x['min'])
        normalized_y = (y_coords - self.normalizer_y['min']) / (self.normalizer_y['max'] - self.normalizer_y['min'])
        normalized_width = (widths - self.normalizer_width['min']) / (self.normalizer_width['max'] - self.normalizer_width['min'])

        return torch.stack([normalized_x, normalized_y, normalized_width], dim=1)

############ EXAMPLE USAGE ############
import argparse

args = argparse.Namespace(
    # Define your arguments here
)

env = CraverDesignEnv(args)

# Print information about the extracted graph
print(f"Number of nodes: {env.torch_graph.num_nodes}")
print(f"Number of edges: {env.torch_graph.num_edges}")
print(f"Node feature shape: {env.torch_graph.x.shape}")
print(f"Edge index shape: {env.torch_graph.edge_index.shape}")
print(f"Edge attribute shape: {env.torch_graph.edge_attr.shape}")

# Initialize the GAT model
in_channels = env.torch_graph.num_node_features
hidden_channels = 64
out_channels = 32
initial_heads = 8
second_heads = 1
edge_dim = env.torch_graph.edge_attr.size(1)
action_hidden_channels = 32
action_dim = 10  # Adjust this based on your requirements
gmm_hidden_dim = 64
num_mixtures = 3

gat_model = GATv2ActorCritic(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    initial_heads=initial_heads,
    second_heads=second_heads,
    edge_dim=edge_dim,
    action_hidden_channels=action_hidden_channels,
    action_dim=action_dim,
    gmm_hidden_dim=gmm_hidden_dim,
    num_mixtures=num_mixtures
)

# Use the GAT model to process the graph
x, edge_index, edge_attr = env.torch_graph.x, env.torch_graph.edge_index, env.torch_graph.edge_attr
batch = torch.zeros(env.torch_graph.num_nodes, dtype=torch.long)  # Assuming a single graph

# Generate crosswalk proposals
proposed_crosswalks, num_actual_proposals, total_log_prob = gat_model.act(x, edge_index, edge_attr, batch)

print(f"\nNumber of actual proposals: {num_actual_proposals}")
print(f"Total log probability of the action: {total_log_prob.item()}")
print("\nProposed crosswalks:")
for i, (location, thickness) in enumerate(proposed_crosswalks):
    if i < num_actual_proposals:
        print(f"  Proposal {i+1}: Location: {location:.4f}, Thickness: {thickness:.2f}")
    else:
        print(f"  Padding {i+1}: Location: {location:.4f}, Thickness: {thickness:.2f}")

# Evaluate the proposed action
state = (x, edge_index, edge_attr, batch)
action = proposed_crosswalks

action_log_probs, state_value, entropy, num_proposals_probs = gat_model.evaluate(state, action)

print(f"\nAction log probabilities shape: {action_log_probs.shape}")
print(f"State value: {state_value.item()}")
print(f"Entropy: {entropy.item()}")
print(f"Number of proposals probabilities shape: {num_proposals_probs.shape}")

# Visualize the GMM distribution
print("\nVisualizing the GMM distribution...")
gmm = gat_model.get_gmm_distribution(gat_model.forward(x, edge_index, edge_attr, batch))
gat_model.visualize_gmm(gmm)

# After initializing the environment, you can print the normalizer values:
print("\nNormalizer values:")
print(f"X-coordinate: {env.normalizer_x}")
print(f"Y-coordinate: {env.normalizer_y}")
print(f"Width: {env.normalizer_width}")
