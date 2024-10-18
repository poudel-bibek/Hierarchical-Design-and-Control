import random
import gymnasium as gym
import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from itertools import tee
from craver_config import (PHASES, DIRECTIONS_AND_EDGES, CONTROLLED_CROSSWALKS_DICT, initialize_lanes, get_phase_groups)

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
        
        self.interconnected_graph = self.construct_interconnected_graph()
        self.visualize_interconnected_graph()
        
    @property
    def action_space(self):
        """
        """
        return gym.spaces.MultiDiscrete([2] * 15)


    @property
    def observation_space(self,):
        """
        """
        return gym.spaces.Box(low=0, high=1, shape=(10, 74), dtype=np.float32)

    def extract_original_graph(self):
        """
        Extracts the pedestrian graph from the original SUMO network file, including walking areas, connections, and junctions.
        """
        

    def step(self, action):
        """
        """
        pass
    

    def _apply_action(self, action):
        """
        """
        pass
    
    def _get_reward(self, action):
        """
        """
        pass

    def reset(self,):
        """
        """
        pass

    def close(self,):
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
    
    def _add_corridor_crosswalks(self,):
        """
        Step 2: Based on the current action representation, add number_of_crosswalks with a thickness_of_crosswalks.
        """
        #number_of_crosswalks, thickness_of_crosswalks  = # some function call/
        pass

    def visualize_pedestrian_graph(self, save_path=None, show=True):
        """
        Visualizes the pedestrian graph with enhanced features, including separate colors for walking areas and junctions.
        """
        plt.figure(figsize=(16, 12))
        
        # Draw walking graph
        pos_walking = {node: node for node in self.walking_graph.nodes()}
        nx.draw_networkx_edges(self.walking_graph, pos_walking, edge_color='orange', width=1, alpha=0.6)
        nx.draw_networkx_nodes(self.walking_graph, pos_walking, node_size=10, node_color='lightblue', alpha=0.6)
        
        # Draw junction graph
        pos_junction = {node: node for node in self.junction_graph.nodes()}
        nx.draw_networkx_nodes(self.junction_graph, pos_junction, node_size=50, node_color='red', alpha=0.8)
        nx.draw_networkx_edges(self.junction_graph, pos_junction, edge_color='green', width=2, alpha=0.6)
        
        plt.title("Pedestrian Graph Visualization", fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def construct_interconnected_graph(self):
        """
        Constructs a graph from the junctions in the network file.
        """
        G = nx.Graph()
        
        tree = ET.parse(self.original_net_file)
        root = tree.getroot()

        # Extract junctions
        for junction in root.findall(".//junction"):
            junction_id = junction.get('id')
            x, y = float(junction.get('x')), float(junction.get('y'))
            junction_type = junction.get('type')
            
            G.add_node(junction_id, pos=(x, y), type=junction_type)

        # Connect junctions based on edges
        for edge in root.findall(".//edge"):
            from_junction = edge.get('from')
            to_junction = edge.get('to')
            if from_junction in G.nodes() and to_junction in G.nodes():
                G.add_edge(from_junction, to_junction)

        return G

    def visualize_interconnected_graph(self, save_path='interconnected_graph.png', show=True):
        """
        Visualizes the interconnected graph of junctions.
        """
        plt.figure(figsize=(20, 15))
        
        pos = nx.get_node_attributes(self.interconnected_graph, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.interconnected_graph, pos, node_size=30, node_color='slateblue', alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.interconnected_graph, pos, edge_color='orange', width=2, alpha=0.2)
        
        # Draw labels
        #nx.draw_networkx_labels(self.interconnected_graph, pos, font_size=8)
        
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

############ EXAMPLE USAGE ############
import argparse

args = argparse.Namespace(
    # Define your arguments here
)

env = CraverDesignEnv(args)


# env.reset()
# env.step(action)
# env.close()





# tree = ET.parse(self.original_net_file)
#         root = tree.getroot()

#         walking_graph = nx.Graph()
#         junction_graph = nx.Graph()

#         # Extract junctions
#         junctions = {}
#         for junction in root.findall(".//junction"):
#             x, y = float(junction.get('x')), float(junction.get('y'))
#             junction_id = junction.get('id')
#             junction_type = junction.get('type')
            
#             node = (x, y)
#             junctions[junction_id] = node
#             junction_graph.add_node(node, pos=node, id=junction_id, junction_type=junction_type)

#         # Extract walking areas
#         walking_areas = {}
#         for edge in root.findall(".//edge[@function='walkingarea']"):
#             edge_id = edge.get('id')
#             for lane in edge.findall('lane'):
#                 shape = lane.get('shape')
#                 coords = [tuple(map(float, point.split(','))) for point in shape.strip().split()]
#                 walking_areas[edge_id] = coords
                
#                 for i in range(len(coords) - 1):
#                     walking_graph.add_edge(coords[i], coords[i+1])

#         # Extract connections between walking areas and junctions
#         for connection in root.findall(".//connection"):
#             from_edge = connection.get('from')
#             to_edge = connection.get('to')
            
#             if from_edge in walking_areas and to_edge in walking_areas:
#                 from_coords = walking_areas[from_edge]
#                 to_coords = walking_areas[to_edge]
                
#                 # Connect the closest points between the two walking areas
#                 closest_pair = min(((f, t) for f in from_coords for t in to_coords), key=lambda x: ((x[0][0]-x[1][0])**2 + (x[0][1]-x[1][1])**2))
#                 walking_graph.add_edge(closest_pair[0], closest_pair[1])

#             # Connect walking areas to junctions
#             if from_edge in walking_areas and to_edge.startswith(':'):
#                 from_coords = walking_areas[from_edge]
#                 to_junction = junctions.get(to_edge[1:])  # Remove ':' prefix
#                 if to_junction:
#                     closest_point = min(from_coords, key=lambda x: ((x[0]-to_junction[0])**2 + (x[1]-to_junction[1])**2))
#                     walking_graph.add_edge(closest_point, to_junction)
#                     junction_graph.add_edge(to_junction, closest_point)

#         # Extract crossings
#         for crossing in root.findall(".//crossing"):
#             node = crossing.get('node')
#             shape = crossing.get('shape')
#             if shape:
#                 coords = [tuple(map(float, point.split(','))) for point in shape.strip().split()]
#                 for i in range(len(coords) - 1):
#                     walking_graph.add_edge(coords[i], coords[i+1])
                
#                 # Connect crossing to junction
#                 if node in junctions:
#                     junction_node = junctions[node]
#                     closest_point = min(coords, key=lambda x: ((x[0]-junction_node[0])**2 + (x[1]-junction_node[1])**2))
#                     walking_graph.add_edge(closest_point, junction_node)
#                     junction_graph.add_edge(junction_node, closest_point)

#         print(f"Walking graph created with {walking_graph.number_of_nodes()} nodes and {walking_graph.number_of_edges()} edges.")
#         print(f"Junction graph created with {junction_graph.number_of_nodes()} nodes and {junction_graph.number_of_edges()} edges.")
#         return walking_graph, junction_graph


"""
def _modify_net_file(self, crosswalks_to_disable):
    # Previous version. check for relevance.
    # Just for changing the appearence of disallowed crosswalks. Not used right now.

    tree = ET.parse(self.original_net_file)
    root = tree.getroot()

    for crosswalk_id in crosswalks_to_disable:
        # Find the edge element corresponding to this crosswalk
        edge = root.find(f".//edge[@id='{crosswalk_id}']")
        if edge is not None:
            # Find the lane within the crosswalk
            lane = edge.find('lane')
            if lane is not None:
                lane.set('width', '0.1')

    tree.write('./SUMO_files/modified_craver_road.net.xml')

# This should be done here before the SUMO call. This can disallow pedestrians before the simulation run.
# Randomly select crosswalks to disable
# to_disable = random.sample(self.controlled_crosswalks, min(5, len(self.controlled_crosswalks)))
# Before sumo call 
# self._modify_net_file(to_disable)

"""
