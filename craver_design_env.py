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
import os
import shutil

def pairwise(iterable):
    """
    Generates consecutive pairs from an iterable.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def clear_folders():
    """
    Clear existing folders (graph_iterations, network_iterations, gmm_plots) if they exist.
    """
    folders_to_clear = ['graph_iterations', 'network_iterations', 'gmm_plots']
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Cleared existing {folder} folder.")
        os.makedirs(folder)
        print(f"Created new {folder} folder.")

class CraverDesignEnv(gym.Env):
    """
    For the higher level agent, modifies the net file based on the design decision.
    No need to connect or close this environment. Will be limited to network file modifications.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.original_net_file = './SUMO_files/original_craver_road.net.xml'

        # Clear existing folders and create new ones
        clear_folders()

        self.crosswalks_to_remove = [crosswalk_id for _, data in CONTROLLED_CROSSWALKS_DICT.items() for crosswalk_id in data['ids']]
        #self._clear_corridor(self.crosswalks_to_remove)
        
        self.original_pedestrian_graph = self._extract_original_graph()
        self.pedestrian_graph = self.original_pedestrian_graph.copy()
        
        if self.args.save_graph_images == 'True':
            self._visualize_pedestrian_graph(save_path='graph_iterations/original_pedestrian_graph.png')
        
        # Initialize normalizer values
        self.normalizer_x = None
        self.normalizer_y = None
        self.normalizer_width = None
        
        # Convert the NetworkX graph to a PyTorch Geometric Data object
        self.torch_graph = self._convert_to_torch_geometric()
        
        self.iteration = 0

    @property
    def action_space(self):
        """
        
        """
        
        return gym.spaces.MultiDiscrete([2] * 15)

    @property
    def observation_space(self):
        """
        
        """

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
        """
        # Update the graph based on the action
        self._update_graph(action)

        # Here you would typically:
        # 1. Apply the action
        # 2. Calculate the reward
        # 3. Determine if the episode is done
        # 4. Collect any additional info for the info dict

        # For now, we'll just return placeholder values
        observation = self.torch_graph
        reward = 0  # You need to implement a proper reward function
        done = False
        info = {}

        self.iteration += 1

        return observation, reward, done, info

    def _apply_action(self, action):
        pass
    
    def _get_reward(self, action):
        pass

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        # Reset the graph to its original state
        self.pedestrian_graph = self.original_pedestrian_graph.copy()
        
        # Recreate the PyTorch Geometric Data object
        self.torch_graph = self._convert_to_torch_geometric()

        self.iteration = 0

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
    
    def _update_graph(self, action):
        """
        Updates the graph based on the action.
        The action is expected to be a list of tuples (location, thickness) for each proposed crosswalk.
        """
        for i, (location, thickness) in enumerate(action):
            # Denormalize the location (x-coordinate) and thickness
            denorm_location = self.normalizer_x['min'] + location * (self.normalizer_x['max'] - self.normalizer_x['min'])
            denorm_thickness = self.normalizer_width['min'] + thickness * (self.normalizer_width['max'] - self.normalizer_width['min'])

            # Find the closest existing nodes (junctions) to place the new crosswalk
            closest_nodes = self._find_closest_nodes(denorm_location)

            if closest_nodes:
                # Create a new node for the crosswalk
                new_node_id = f"crosswalk_{self.iteration}_{i}"
                y_coord = sum(self.pedestrian_graph.nodes[node]['pos'][1] for node in closest_nodes) / len(closest_nodes)
                self.pedestrian_graph.add_node(new_node_id, pos=(denorm_location, y_coord), type='crosswalk', width=denorm_thickness)

                # Connect the new node to the closest existing nodes
                for node in closest_nodes:
                    self.pedestrian_graph.add_edge(new_node_id, node, width=denorm_thickness)

        # After updating the graph, we need to update the PyTorch Geometric Data object
        self.torch_graph = self._convert_to_torch_geometric()

        if self.args.save_network_xml:
            self._save_graph_as_xml(f'updated_network_iteration_{self.iteration}.net.xml')

        if self.args.save_graph_images:
            self._save_graph(f'pedestrian_graph_iteration_{self.iteration}.png')

    def _find_closest_nodes(self, x_location):
        """
        Finds the closest existing nodes (junctions) to the given x_location.
        Returns a list of node IDs.
        """
        nodes = list(self.pedestrian_graph.nodes(data=True))
        sorted_nodes = sorted(nodes, key=lambda n: abs(n[1]['pos'][0] - x_location))
        
        # Get the two closest nodes
        closest_nodes = [sorted_nodes[0][0], sorted_nodes[1][0]]
        
        return closest_nodes

    def _visualize_pedestrian_graph(self, save_path='pedestrian_graph.png'):
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

    def _save_graph(self, filename):
        """
        Saves the current state of the graph as an image.
        """
        plt.figure(figsize=(20, 15))
        
        pos = nx.get_node_attributes(self.pedestrian_graph, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.pedestrian_graph, pos, node_size=30, node_color='slateblue', alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.pedestrian_graph, pos, edge_color='orange', width=2, alpha=0.2)
        
        plt.title(f"Pedestrian Graph - Iteration {self.iteration}", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Create a 'graph_iterations' directory if it doesn't exist
        os.makedirs('graph_iterations', exist_ok=True)
        
        # Save the graph in the 'graph_iterations' directory
        plt.savefig(os.path.join('graph_iterations', filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_graph_as_xml(self, filename):
        """
        Saves the current state of the graph as a SUMO network XML file.
        """
        # Load the original XML file
        tree = ET.parse(self.original_net_file)
        root = tree.getroot()

        # Remove existing crosswalks
        for crosswalk in root.findall(".//crossing"):
            root.remove(crosswalk)

        # Add new crosswalks based on the updated graph
        for node, data in self.pedestrian_graph.nodes(data=True):
            if data['type'] == 'crosswalk':
                # Find non-crosswalk neighbors
                non_crosswalk_neighbors = [j for j in self.pedestrian_graph.neighbors(node) 
                                           if self.pedestrian_graph.nodes[j]['type'] != 'crosswalk']
                
                if non_crosswalk_neighbors:
                    # Find the nearest junction
                    nearest_junction = min(
                        non_crosswalk_neighbors,
                        key=lambda j: ((data['pos'][0] - self.pedestrian_graph.nodes[j]['pos'][0])**2 +
                                       (data['pos'][1] - self.pedestrian_graph.nodes[j]['pos'][1])**2)**0.5
                    )

                    # Create a new crossing element
                    crossing = ET.SubElement(root, 'crossing')
                    crossing.set('id', node)
                    crossing.set('node', nearest_junction)
                    crossing.set('width', str(data['width']))
                    crossing.set('shape', f"{data['pos'][0]},{data['pos'][1]} {data['pos'][0]},{data['pos'][1]}")  # Simplified shape
                else:
                    print(f"Warning: Crosswalk {node} has no non-crosswalk neighbors. Skipping this crosswalk in XML output.")

        # Create a 'network_iterations' directory if it doesn't exist
        os.makedirs('network_iterations', exist_ok=True)

        # Save the updated XML file
        tree.write(os.path.join('network_iterations', filename))
        print(f"Updated network saved to {os.path.join('network_iterations', filename)}")

############ EXAMPLE USAGE ############
import argparse

parser = argparse.ArgumentParser(description="CraverDesignEnv arguments")
parser.add_argument('--save_graph_images', type=str, choices=['True', 'False'], default='True', help='Save graph images')
parser.add_argument('--save_network_xml', type=str, choices=['True', 'False'], default='True', help='Save network XML files')
parser.add_argument('--save_gmm_plots', type=str, choices=['True', 'False'], default='True', help='Save GMM distribution plots')
args = parser.parse_args()

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

# Run through 5 iterations
for iteration in range(5):
    print(f"\n--- Iteration {iteration + 1} ---")
    
    # Use the GAT model to process the graph
    x, edge_index, edge_attr = env.torch_graph.x, env.torch_graph.edge_index, env.torch_graph.edge_attr
    batch = torch.zeros(env.torch_graph.num_nodes, dtype=torch.long)  # Assuming a single graph

    # Generate crosswalk proposals
    proposed_crosswalks, num_actual_proposals, total_log_prob = gat_model.act(x, edge_index, edge_attr, batch)

    print(f"Number of actual proposals: {num_actual_proposals}")
    print(f"Total log probability of the action: {total_log_prob.item()}")
    print("Proposed crosswalks:")
    for i, (location, thickness) in enumerate(proposed_crosswalks):
        if i < num_actual_proposals:
            print(f"  Proposal {i+1}: Location: {location:.4f}, Thickness: {thickness:.2f}")

    # Evaluate the proposed action
    state = (x, edge_index, edge_attr, batch)
    action = proposed_crosswalks

    action_log_probs, state_value, entropy, num_proposals_probs = gat_model.evaluate(state, action)

    print(f"State value: {state_value.item()}")
    print(f"Entropy: {entropy.item()}")

    # Visualize and save the GMM distribution
    if args.save_gmm_plots == 'True':
        print("Saving GMM distribution plot...")
        gmm = gat_model.get_gmm_distribution(gat_model.forward(x, edge_index, edge_attr, batch))
        os.makedirs('gmm_plots', exist_ok=True)
        gat_model.visualize_gmm(gmm, save_path=f'gmm_plots/gmm_distribution_iteration_{iteration}.png')

    # Update the environment
    observation, reward, done, info = env.step(proposed_crosswalks)

    if done:
        print("Environment signaled done. Resetting...")
        env.reset()

# After running all iterations, print the final normalizer values:
print("\nFinal Normalizer values:")
print(f"X-coordinate: {env.normalizer_x}")
print(f"Y-coordinate: {env.normalizer_y}")
print(f"Width: {env.normalizer_width}")
