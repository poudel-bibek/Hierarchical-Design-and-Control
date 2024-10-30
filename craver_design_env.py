import random
import gymnasium as gym
import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
from itertools import tee
from craver_config import (PHASES, DIRECTIONS_AND_EDGES, CONTROLLED_CROSSWALKS_DICT, initialize_lanes)
import torch
from torch_geometric.data import Data
from models import GATv2ActorCritic  # Import the GAT model
import os
import shutil
from gymnasium import spaces
from utils import *

"""
For normalizers, 
analyse the entire graph to get the min and max values?


"""
def pairwise(iterable):
    """
    Generates consecutive pairs from an iterable.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def clear_folders():
    """
    Clear existing folders (graph_iterations, network_iterations, gmm_iterations) if they exist.
    """
    folders_to_clear = ['graph_iterations', './SUMO_files/network_iterations', 'gmm_iterations']
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_proposals = config['max_proposals']
        self.min_thickness = config['min_thickness']
        self.max_thickness = config['max_thickness']
        self.min_coordinate = config['min_coordinate']
        self.max_coordinate = config['max_coordinate']
        self.original_net_file = config['original_net_file']

        # Clear existing folders and create new ones
        clear_folders()

        # The crosswalks present initially. 
        self.crosswalks_to_remove = [crosswalk_id for _, data in CONTROLLED_CROSSWALKS_DICT.items() for crosswalk_id in data['ids']]
        #self._clear_corridor(self.crosswalks_to_remove)
        
        self.original_pedestrian_graph = self._extract_original_graph()
        self.iterative_pedestrian_graph = self._cleanup_corridor(self.original_pedestrian_graph)

        # Initialize normalizer values before visualization
        self._initialize_normalizers()

        if self.config.get('save_graph_images', False):
            save_graph_visualization(graph=self.original_pedestrian_graph, iteration='ORIGINAL')
            save_graph_visualization(graph=self.iterative_pedestrian_graph, iteration='BASE_GRAPH')
            save_better_graph_visualization(graph=self.iterative_pedestrian_graph, iteration='BASE_GRAPH')
            save_better_graph_visualization(graph=self.original_pedestrian_graph, iteration='ORIGINAL')
        # Initialize normalizer values
        self.normalizer_x = None
        self.normalizer_y = None
        self.normalizer_width = self.max_thickness

    @property
    def action_space(self):
        """
        Defines the action space for the environment.
        """
        return spaces.Dict({
            'num_proposals': spaces.Discrete(self.max_proposals + 1),  # 0 to max_proposals
            'proposals': spaces.Box(
                low=np.array([[self.min_coordinate, self.min_thickness]] * self.max_proposals),
                high=np.array([[self.max_coordinate, self.max_thickness]] * self.max_proposals),
                dtype=np.float32
            )
        })

    @property
    def observation_space(self):
        """
        Returns an arbitrary high-dimensional shape for the observation space.
        Note: This is an arbitrary shape and doesn't reflect the actual dimensions of the graph.
        The GATv2 model can handle variable-sized inputs, so this is mainly for compatibility.
        """
        return spaces.Box(low=0, high=1, shape=(1000, 3), dtype=np.float32)

    def _extract_original_graph(self):
        """
        Extracts the pedestrian graph from the original SUMO network file, including walking areas, connections, and junctions.
        For our purposes:
        - What makes a node?
            - Junctions
            - Attributes: pos (x, y)
        - What makes an edge?
            - Walking areas between junctions
            - Attributes: width

        # After getting edges, remove the nodes that dont have edges.
        """

        G = nx.Graph()
        tree = ET.parse(self.original_net_file)
        root = tree.getroot()
        # print(f"tree: {tree}, \nroot: \ntag: {root.tag}, \nattrib: {root.attrib}")
        # for child in root:
        #     print(f"child: {child.tag}, attrib: {child.attrib}")

        # Extract junctions
        for junction in root.findall(".//junction"):
            junction_id = junction.get('id')
            x, y = float(junction.get('x')), float(junction.get('y'))
            G.add_node(junction_id, pos=(x, y))

        # only getting edges that have function="walkingarea" does not work. Perhaps the function could be internal. 
        # For each edge, go to the lane and check if allow="pedestrian" exists
        # Connect junctions based on edges with pedestrian lanes
        for edge in root.findall(".//edge"):
            from_junction = edge.get('from')
            to_junction = edge.get('to')

            if from_junction in G.nodes() and to_junction in G.nodes():

                # Check if any lane allows pedestrians
                for lane in edge.findall('lane'):
                    allow = lane.get('allow', '')
                    
                    if 'pedestrian' in allow:
                        width = float(lane.get('width', 0.0))
                        G.add_edge(from_junction, to_junction, width=width)
                        break  # Only need to add edge once if multiple pedestrian lanes

        # Remove nodes that dont have edges
        G.remove_nodes_from(list(nx.isolates(G)))

        # Add debug print at the end
        print("\nGraph Coordinate Ranges:")
        pos = nx.get_node_attributes(G, 'pos')
        x_coords = [coord[0] for coord in pos.values()]
        y_coords = [coord[1] for coord in pos.values()]
        print(f"X range: {min(x_coords):.2f} to {max(x_coords):.2f}")
        print(f"Y range: {min(y_coords):.2f} to {max(y_coords):.2f}")
        
        return G

    def step(self, action, iteration):
        """
        Implement the environment step here.
        """
        print(f"Action received: {action}")
        # Convert tensor action to proposals
        action = action.cpu().numpy()  # Convert to numpy array if it's not already
        num_proposals = np.count_nonzero(action.any(axis=1))  # Count non-zero rows
        proposals = action[:num_proposals]  # Only consider the actual proposals

        # Apply the action to update the graph
        self._apply_action(proposals, iteration)

        # Here you would typically:
        # 1. Calculate the reward
        # 2. Determine if the episode is done
        # 3. Collect any additional info for the info dict

        observation = self.iterative_torch_graph
        reward = 0  # You need to implement a proper reward function
        done = False
        info = {}

        return observation, reward, done, info

    def _apply_action(self, proposals, iteration):
        """
        Updates the graph based on the action.
        The proposals are expected to be a list of tuples (location, thickness) for each proposed crosswalk.

        When a location is proposed, its actually the location of where an edge should be added.
        i.e., corresponding node positions along the corridor should be found and connected.
        """
        for i, (location, thickness) in enumerate(proposals):
            # Denormalize the location (x-coordinate) and thickness
            denorm_location = self.normalizer_x['min'] + location * (self.normalizer_x['max'] - self.normalizer_x['min'])
            denorm_thickness = thickness * self.normalizer_width

            # Find the closest existing nodes (junctions) to place the new crosswalk
            closest_nodes = self._find_closest_nodes(denorm_location)

            if closest_nodes:
                # Create a new node for the crosswalk
                new_node_id = f"crosswalk_{iteration}_{i}"
                y_coord = sum(self.iterative_pedestrian_graph.nodes[node]['pos'][1] for node in closest_nodes) / len(closest_nodes)
                self.iterative_pedestrian_graph.add_node(new_node_id, pos=(denorm_location, y_coord))

                # Connect the new node to the closest existing nodes
                for node in closest_nodes:
                    self.iterative_pedestrian_graph.add_edge(new_node_id, node, width=denorm_thickness)

        # After updating the graph, we need to update the PyTorch Geometric Data object
        self.iterative_torch_graph = self._convert_to_torch_geometric()

        if self.config.get('save_network_xml', False):
            self._save_graph_as_xml(iteration)

        if self.config.get('save_graph_images', False):
            save_graph_visualization(self.iterative_pedestrian_graph, iteration)

    def reset(self):
        """
        Reset the environment to its initial state.
        Should return x, edge_index, edge_attr, batch

        Do we want the design agent to ever see the original graph?
        """

        # Reset the graph to its original state
        self.iterative_pedestrian_graph = self.original_pedestrian_graph.copy()
        
        # Recreate the PyTorch Geometric Data object
        self.iterative_torch_graph = self._convert_to_torch_geometric()
        
        # nodes, edge_index, edge_attr 
        return self.iterative_torch_graph

    def close(self):
        """
        
        """
        pass
    
    def _cleanup_corridor(self, graph):
        """
        Cleanup the corridor.
        1. Do we want the design agent to ever see the original graph?
            - No because we will be benchmarking against it. 
            - Hence clear the corridor of all existing crosswalks.
        
        2. Remove nodes and edges too far away from the corridor (based on y values).

        3. Remove some fringe nodes. That exist because some vehicle roads allow pedestrians.
        This step creates a base graph upon which additional edges are added.
        This step requires some manual input.
        """

        cleanup_graph = graph.copy()
        print(f"\nBefore cleanup: {len(cleanup_graph.nodes())} nodes, {len(cleanup_graph.edges())} edges\n")
        
        # Nodes to remove
        crosswalk_nodes = ['9727816950', '9727816844', '9727816623', 'cluster_9740157181_9740483933', '9740157192', 
                           '9740484527', 'cluster_9740157181_9740483933','9727816850', 'cluster_9740411700_9740411702','9740157153']
        fringe_nodes = ['9727779406', '9740484031', '9740155241', '9740157194', '9740157209', '9740484521', '9740484518','9740157155']
        nodes_to_remove = crosswalk_nodes + fringe_nodes
        
        # Remove the specified nodes and their associated edges
        cleanup_graph.remove_nodes_from(nodes_to_remove)
        
        # Remove any isolated nodes that might have been created
        cleanup_graph.remove_nodes_from(list(nx.isolates(cleanup_graph)))
        
        # remove the edges associated with the removed nodes
        cleanup_graph.remove_edges_from(cleanup_graph.edges(nodes_to_remove))
        
        print(f"\nAfter cleanup: {len(cleanup_graph.nodes())} nodes, {len(cleanup_graph.edges())} edges\n")
        
        return cleanup_graph
    
    def _find_closest_nodes(self, x_location):
        """
        Finds the closest existing nodes (junctions) to the given x_location.
        Returns a list of node IDs.
        """
        nodes = list(self.iterative_pedestrian_graph.nodes(data=True))
        sorted_nodes = sorted(nodes, key=lambda n: abs(n[1]['pos'][0] - x_location))
        
        # Get the two closest nodes
        closest_nodes = [sorted_nodes[0][0], sorted_nodes[1][0]]
        
        return closest_nodes
    
    def _convert_to_torch_geometric(self):
        """
        Converts the NetworkX graph to a PyTorch Geometric Data object.
        Normalizes the coordinates to lie between 0 and 1 and scales the width values proportionally.
        """
        # Extract node features (x, y coordinates)
        node_features = []
        for node, data in self.iterative_pedestrian_graph.nodes(data=True):
            node_features.append([data['pos'][0], data['pos'][1]])

        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Normalize features
        x = self._normalize_features(x)
        
        # Extract edge indices and attributes
        edge_index = []
        edge_attr = []
        for edge in self.iterative_pedestrian_graph.edges(data=True):
            source = list(self.iterative_pedestrian_graph.nodes()).index(edge[0])
            target = list(self.iterative_pedestrian_graph.nodes()).index(edge[1])
            edge_index.append([source, target])
            edge_index.append([target, source])  # Add reverse edge for undirected graph
            
            width = edge[2]['width']
            # Normalize edge width
            normalized_width = width / self.normalizer_width
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
        Normalize the features (coordinates) to lie between 0 and 1.
        Save the normalizer values for potential later use.
        """
        x_coords = features[:, 0]
        y_coords = features[:, 1]

        # Compute normalizers
        self.normalizer_x = {'min': x_coords.min(), 'max': x_coords.max()}
        self.normalizer_y = {'min': y_coords.min(), 'max': y_coords.max()}

        # Normalize features
        normalized_x = (x_coords - self.normalizer_x['min']) / (self.normalizer_x['max'] - self.normalizer_x['min'])
        normalized_y = (y_coords - self.normalizer_y['min']) / (self.normalizer_y['max'] - self.normalizer_y['min'])

        return torch.stack([normalized_x, normalized_y], dim=1)
    
    def _save_graph_as_xml(self, iteration):
        """
        Saves the current state of the graph in netrowkx as a SUMO network XML file.
        when saving, changes made from previous iterations are discarded. i.e., only the changes of this iteration on the base graph are saved.
        """

        filename = f'network_iteration_{iteration}.net.xml' 

        # Load the original XML file
        tree = ET.parse(self.original_net_file)
        root = tree.getroot()

        # Remove existing crosswalks
        for crosswalk in root.findall(".//crossing"):
            root.remove(crosswalk)

        # Add new crosswalks based on the updated graph
        for node, data in self.iterative_pedestrian_graph.nodes(data=True):

            # Find non-crosswalk neighbors
            non_crosswalk_neighbors = [j for j in self.iterative_pedestrian_graph.neighbors(node)]
            
            if non_crosswalk_neighbors:
                # Find the nearest junction
                nearest_junction = min(
                    non_crosswalk_neighbors,
                    key=lambda j: ((data['pos'][0] - self.iterative_pedestrian_graph.nodes[j]['pos'][0])**2 +
                                    (data['pos'][1] - self.iterative_pedestrian_graph.nodes[j]['pos'][1])**2)**0.5
                )

                # Create a new crossing element
                crossing = ET.SubElement(root, 'crossing')
                crossing.set('id', node)
                crossing.set('node', nearest_junction)
                crossing.set('shape', f"{data['pos'][0]},{data['pos'][1]} {data['pos'][0]},{data['pos'][1]}")  # Simplified shape


        # Create a 'network_iterations' directory if it doesn't exist
        os.makedirs('./SUMO_files/network_iterations', exist_ok=True)

        # Save the updated XML file
        tree.write(os.path.join('./SUMO_files/network_iterations', filename))
        print(f"Updated network saved to {os.path.join('./SUMO_files/network_iterations', filename)}")

    def _initialize_normalizers(self):
        """
        Initialize normalizers based on the graph coordinates
        """
        # Extract all x and y coordinates from the graph
        coords = np.array([data['pos'] for _, data in self.original_pedestrian_graph.nodes(data=True)])
        
        if len(coords) > 0:
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            
            self.normalizer_x = {'min': float(np.min(x_coords)), 'max': float(np.max(x_coords))}
            self.normalizer_y = {'min': float(np.min(y_coords)), 'max': float(np.max(y_coords))}
        else:
            # Fallback values if the graph is empty
            self.normalizer_x = {'min': 0.0, 'max': 1.0}
            self.normalizer_y = {'min': 0.0, 'max': 1.0}

############ EXAMPLE USAGE ############
# import argparse
# parser = argparse.ArgumentParser(description="CraverDesignEnv arguments")
# parser.add_argument('--save_graph_images', action='store_true', help='Save graph images')
# parser.add_argument('--save_network_xml', action='store_true', help='Save network XML files')
# parser.add_argument('--save_gmm_plots', action='store_true', help='Save GMM distribution plots')
# parser.add_argument('--max_proposals', type=int, default=10, help='Maximum number of crosswalk proposals')
# parser.add_argument('--min_thickness', type=float, default=0.1, help='Minimum thickness of crosswalks')
# parser.add_argument('--max_thickness', type=float, default=1.0, help='Maximum thickness of crosswalks')
# parser.add_argument('--min_coordinate', type=float, default=0.0, help='Minimum coordinate for crosswalk placement')
# parser.add_argument('--max_coordinate', type=float, default=1.0, help='Maximum coordinate for crosswalk placement')
# args = parser.parse_args()

"""
design_args = {
    'save_graph_images': True,
    'save_network_xml': True,
    'save_gmm_plots': True,
    'max_proposals': 10,
    'min_thickness': 0.1,
    'max_thickness': 10.0,
    'min_coordinate': 0.0,
    'max_coordinate': 1.0,
}

env = CraverDesignEnv(design_args)

# Print information about the extracted graph and action space
print(f"Number of nodes: {env.iterative_torch_graph.num_nodes}")
print(f"Number of edges: {env.iterative_torch_graph.num_edges}")
print(f"Node feature shape: {env.iterative_torch_graph.x.shape}")
print(f"Edge index shape: {env.iterative_torch_graph.edge_index.shape}")
print(f"Edge attribute shape: {env.iterative_torch_graph.edge_attr.shape}")
print(f"Action space: {env.action_space}")

# Initialize the GAT model
in_channels = env.iterative_torch_graph.num_node_features
hidden_channels = 64
out_channels = 32
initial_heads = 8
second_heads = 1
edge_dim = env.iterative_torch_graph.edge_attr.size(1)
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
    x, edge_index, edge_attr = env.iterative_torch_graph.x, env.iterative_torch_graph.edge_index, env.iterative_torch_graph.edge_attr
    batch = torch.zeros(env.iterative_torch_graph.num_nodes, dtype=torch.long)  # Assuming a single graph

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
    if design_args.get('save_gmm_plots', False):
        print("Saving GMM distribution plot...")
        gmm = gat_model.get_gmm_distribution(gat_model.forward(x, edge_index, edge_attr, batch))
        os.makedirs('gmm_iterations', exist_ok=True)
        gat_model.visualize_gmm(gmm, save_path=f'gmm_iterations/gmm_distribution_iteration_{iteration}.png')

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
"""
