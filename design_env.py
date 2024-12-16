import os
import math 
import random
import shutil
from itertools import tee

import wandb
wandb.require("core") # Bunch of improvements in using the core.
import subprocess
import gymnasium as gym
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as mp
import xml.etree.ElementTree as ET
from gymnasium import spaces
from torch_geometric.data import Data

import traci
import queue
from ppo_alg import PPO, Memory
from control_env import ControlEnv
from sim_config import (PHASES, DIRECTIONS_AND_EDGES, 
                       CONTROLLED_CROSSWALKS_DICT, initialize_lanes)
from utils import *
from models import CNNActorCritic

def parallel_worker(rank, control_args, model_init_params, policy_old_dict, memory_queue, global_seed, worker_device, network_iteration):
    """
    At every iteration, a number of workers will each parallelly carry out one episode in control environment.
    - Worker environment runs in CPU (SUMO runs in CPU).
    - Worker policy inference runs in GPU.
    - memory_queue is used to store the memory of each worker and send it back to the main process.
    - A shared policy_old (dict copy passed here) is used for importance sampling.
    """

    shared_policy_old = CNNActorCritic(model_init_params['model_dim'], model_init_params['action_dim'], **model_init_params['kwargs'])
    shared_policy_old.load_state_dict(policy_old_dict)

    # Set seed for this worker
    worker_seed = global_seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    lower_env = ControlEnv(control_args, worker_id=rank, network_iteration=network_iteration)
    memory_transfer_freq = control_args['memory_transfer_freq']  # Get from config

    # The central memory is a collection of memories from all processes.
    # A worker instance must have their own memory 
    local_memory = Memory()
    shared_policy_old = shared_policy_old.to(worker_device)

    state, _ = lower_env.reset()
    ep_reward = 0
    steps_since_update = 0
    
    for _ in range(control_args['total_action_timesteps_per_episode']):
        state_tensor = torch.FloatTensor(state).to(worker_device)

        # Select action
        with torch.no_grad():
            action, logprob = shared_policy_old.act(state_tensor)
            action = action.cpu()  # Explicitly Move to CPU, Incase they were on GPU
            logprob = logprob.cpu() 

        print(f"\nAction: in worker {rank}: {action}")
        # Perform action
        # These reward and next_state are for the action_duration timesteps.
        next_state, reward, done, truncated, info = lower_env.step(action)
        ep_reward += reward

        # Store data in memory
        local_memory.append(torch.FloatTensor(state), action, logprob, reward, done)
        steps_since_update += 1

        if steps_since_update >= memory_transfer_freq or done or truncated:
            # Put local memory in the queue for the main process to collect
            memory_queue.put((rank, local_memory))
            local_memory = Memory()  # Reset local memory
            steps_since_update = 0

        if done or truncated:
            break

        state = next_state.flatten()

    # In PPO, we do not make use of the total reward. We only use the rewards collected in the memory.
    print(f"Worker {rank} finished. Total reward: {ep_reward}")
    lower_env.close()
    memory_queue.put((rank, None))  # Signal that this worker is done

def pairwise(iterable):
    """
    Generates consecutive pairs from an iterable.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def clear_folders(component_dir, network_dir):
    """
    Clear existing folders (graph_iterations, network_iterations, gmm_iterations) if they exist.
    Create new ones.
    """
    folders_to_clear = ['graph_iterations', network_dir, component_dir, 'gmm_iterations']
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Cleared existing {folder} folder.")
        os.makedirs(folder)
        print(f"Created new {folder} folder.")

class DesignEnv(gym.Env):
    """
    Higher level.
    - Modifies the net file based on design action.
    - No need to connect to or close this environment. 
    - Internally calls the parallel workers of the control environment.
    - Thickness does not need normalization/ denormalization (the sampling process itself proposed between min and max)

    Improved version:
    - During initialization: 
        - Extract the pedestrian walkway graph in networkx and other plain XML components (node, edge, connection, type, tllogic) from the original net file.
        - Extract the normalizers from the original net file.
        - Create a base canvas (networkx and plain XML) by removing fringe, isolated nodes, and existing crosswalks from the networkx graph.
        - After every update to the networkx graph, the plain XML components are updated.
        
    - During reset: 
        - Every iteration, crosswalks are added to the base canvas networkx graph.
        - During reset, add the initial crosswalks (present in the real-world network) to the base canvas.
        - Whenever there is a requirement to return the state (e.g., reset, step), the networkx graph is converted to torch geometric and then state returned.
        
    - During step:
        - Apply action:
            - Denormalize proposed locations (thickness does not need denormalization) and add to networkx graph.
            - Use the same mechanism as reset to update the plain XML components.
    - Note: 
        - XML component files contain things other than the pedestrian network as well (e.g., vehicle lanes, junctions, etc.)
        - Original means unmodified SUMO net file.
        - Base means the networkx graph and plain XML components at the start of the training (before any iterations).
        - Iterative means the networkx graph and plain XML components after every iteration.
    """
    
    def __init__(self, design_args, control_args, lower_ppo_args, is_sweep=False, is_eval=False):
        super().__init__()
        self.control_args = control_args
        self.design_args = design_args
        self.lower_ppo_args = lower_ppo_args
        self.is_sweep = is_sweep
        self.is_eval = is_eval

        self.max_proposals = design_args['max_proposals']
        self.min_thickness = design_args['min_thickness']
        self.min_coordinate = design_args['min_coordinate']
        self.max_coordinate = design_args['max_coordinate']
        self.max_thickness = design_args['max_thickness']
        self.component_dir = design_args['component_dir']
        self.network_dir = design_args['network_dir']
        clear_folders(self.component_dir, self.network_dir) # Do not change the position of this.
        
        # Generate the 5 different component XML files (node, edge, connection, type, tllogic) from the net file.
        self._create_component_xml_files(self.design_args['original_net_file'])

        # Extract networkx graph from the component files. (Also update locations of nodes in existing_crosswalks)
        pedestrian_networkx_graph  = self._extract_networkx_graph() 
        self._initialize_normalizers(pedestrian_networkx_graph) # normalizers are extracted from networkx graph
        self.existing_crosswalks = self._get_existing_crosswalks(pedestrian_networkx_graph) # Make use of the networkx graph to add locations
        
        # Cleanup the pedestrian walkway graph (i.e, remove isolated, fringe, existing crosswalks) to create a base canvas
        self.base_networkx_graph = self._cleanup_graph(pedestrian_networkx_graph, self.existing_crosswalks)
        self.horizontal_nodes_top_ped = ['9666242268', '9666274719', '9666274722', '9666274721', '9727816851', '9666274744', '9666274574', '9666274798', '9666274635', '9666274616', '9666274886', '9655154530', '9655154527', '9655154520', '10054309033', '9740157195', '9740157210', '10054309051', '9740484524', '9740484531']
        self.horizontal_nodes_bottom_ped = ['9727816638', '9727816862', '9727816846', '9727816629', '9727779405', '9740157080', '9727816625', '9740157142', '9740157169', '9740157145', '9740484033', '9740157174', '9740157171', '9740157154', '9740157158', '9740411703', '9740411701', '9740483978', '9740483934', '9740157180', '9740483946', '9740157204', '9740484420', '9740157211', '9740484523', '9740484522', '9740484512', '9740484528', '9739966899', '9739966895']
        self.horizontal_edges_top_veh = ['-16666012#2', '-16666012#3', '-16666012#4', '-16666012#5', '-16666012#6', '-16666012#7', '-16666012#9', '-16666012#11', '-16666012#12', '-16666012#13', '-16666012#14', '-16666012#15', '-16666012#16', '-16666012#17']
        self.horizontal_edges_bottom_veh = ['16666012#2', '16666012#3', '16666012#4', '16666012#5', '16666012#6', '16666012#7', '16666012#9', '16666012#11', '16666012#12', '16666012#13', '16666012#14', '16666012#15', '16666012#16', '16666012#17']

        self._update_xml_files(self.base_networkx_graph, 'base') # Create base XML files from latest networkx graph

        if self.design_args['save_graph_images']:
            save_graph_visualization(graph=pedestrian_networkx_graph, iteration='original')
            save_graph_visualization(graph=self.base_networkx_graph, iteration='base')
            save_better_graph_visualization(graph=pedestrian_networkx_graph, iteration='original')
            save_better_graph_visualization(graph=self.base_networkx_graph, iteration='base')

        # Lower level agent
        self.lower_ppo = PPO(**self.lower_ppo_args)
        self.action_timesteps = 0 # keep track of how many times action has been taken by all lower level workers
        self.writer = self.control_args['writer']
        self.best_reward_lower = float('-inf')

        # Bugfix: Removing unpicklable object (writer) from control_args
        self.control_args_worker = {k: v for k, v in self.control_args.items() if k != 'writer'}
        self.model_init_params_worker = {'model_dim': self.lower_ppo.policy.in_channels,
                                     'action_dim': self.lower_ppo.action_dim,
                                     'kwargs': self.lower_ppo_args['model_kwargs']}

    @property
    def action_space(self):
        """
        
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

    def _get_existing_crosswalks(self, networkx_graph):
        """
        Extract the crosswalks present initially in the XML. (exclude 0, 1, 2, 3, 10)
        Add the node locations to the existing crosswalk from the networkx graph.
        """
        excluded_ids = [0, 1, 2, 10]
        existing_crosswalks = {}
        
        for key, data in CONTROLLED_CROSSWALKS_DICT.items():
            for crosswalk_id in data['ids']:
                if key not in excluded_ids:
                    existing_crosswalks[crosswalk_id] = {
                        'pos': [], # will be added later
                        'crossing_nodes': data['crossing_nodes']
                    }

        for crosswalk_id, crosswalk_data in existing_crosswalks.items():
            for node in crosswalk_data['crossing_nodes']:
                if node in networkx_graph.nodes():
                    crosswalk_data['pos'].append(networkx_graph.nodes[node]['pos'])

        return existing_crosswalks

    def _extract_networkx_graph(self,):
        """
        Extract the pedestrian walkway graph in networkx from the component files.

        To create a pedestrian network: 
        - If a node has an edge with type attribute to 'highway.footway' or 'highway.steps' and allow attribute to include 'pedestrian', keep the node and the edge.
        """
        G = nx.Graph() # undirected graph

        # Parse node file
        node_tree = ET.parse(f'{self.component_dir}/original.nod.xml')
        node_root = node_tree.getroot()

        # Add all nodes first (we'll remove non-pedestrian nodes later)
        for node in node_root.findall('node'):
            node_id = node.get('id')
            x = float(node.get('x'))
            y = float(node.get('y'))
            G.add_node(node_id, pos=(x, y), type='regular')

        # Parse edge file
        edge_tree = ET.parse(f'{self.component_dir}/original.edg.xml')
        edge_root = edge_tree.getroot()
        
        # Keep track of nodes that are part of pedestrian paths
        pedestrian_nodes = set()
        
        # Add edges that are pedestrian walkways
        for edge in edge_root.findall('edge'):
            edge_type = edge.get('type')
            allow = edge.get('allow', '')
            
            # Check if edge is a pedestrian walkway
            if edge_type in ['highway.footway', 'highway.steps'] and 'pedestrian' in allow:
                from_node = edge.get('from')
                to_node = edge.get('to')
                
                if from_node is not None and to_node is not None:
                    # Add edge with its attributes
                    width = float(edge.get('width', 2.0)) # default width is 2.0
                    G.add_edge(from_node, to_node, id=edge.get('id'), width=width)
                    
                    # Mark these nodes as part of pedestrian network
                    pedestrian_nodes.add(from_node)
                    pedestrian_nodes.add(to_node)
        
        # Remove nodes that aren't part of any pedestrian path
        non_pedestrian_nodes = set(G.nodes()) - pedestrian_nodes
        G.remove_nodes_from(non_pedestrian_nodes)
        return G

    def _create_component_xml_files(self, sumo_net_file):
        """
        Creates the base SUMO files (5 files).
        """
        # Node (base_xml.nod.xml), Edge (base_xml.edg.xml), Connection (base_xml.con.xml), Type file (base_xml.typ.xml) and Traffic Light (base_xml.tll.xml)
        # Create the output directory if it doesn't exist
        output_dir = "./SUMO_files/component_SUMO_files"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run netconvert with output files in the specified directory
        command = f"netconvert --sumo-net-file {sumo_net_file} --plain-output-prefix {output_dir}/original --plain-output.lanes true"

        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            if result.stderr:
                print("Warnings/Errors from netconvert:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running netconvert: {e}")
            print("Error output:", e.stderr)

    def step(self, action, num_actual_proposals, iteration, global_step):
        """
        Every step in the design environment involves:
        - Updating the network xml file based on the design action.
        - A number of parallel workers (that utilize the new network file) to each carry out one episode in the control environment.
        """

        # First complete the higher level agent's step.
        #print(f"Action received: {action}")

        # Convert tensor action to proposals
        action = action.cpu().numpy()  # Convert to numpy array if it's not already
        proposals = action.squeeze(0)[:num_actual_proposals]  # Only consider the actual proposals

        #print(f"\n\nProposals: {proposals}\n\n")

        # Apply the action to output the latest SUMO network file as well as modify the iterative_torch_graph.
        self._apply_action(proposals, iteration)

        # Here you would typically:
        # 1. Calculate the reward
        # 2. Determine if the episode is done
        # 3. Collect any additional info for the info dict

        reward = 0  # You need to implement a proper reward function
        done = False
        info = {}

        # Then, for the lower level agent.
        manager = mp.Manager()
        memory_queue = manager.Queue()
        processes = []
        
        for rank in range(self.control_args['lower_num_processes']):
            p = mp.Process(
                target=parallel_worker,
                args=(
                    rank,
                    self.control_args_worker,
                    self.model_init_params_worker,
                    self.lower_ppo.policy_old.state_dict(),
                    memory_queue,
                    self.control_args['global_seed'],
                    self.lower_ppo_args['device'],
                    iteration
                )
            )
            p.start()
            processes.append(p)

        if self.control_args['lower_anneal_lr']:
            current_lr = self.lower_ppo.update_learning_rate(iteration)

        all_memories = []
        active_workers = set(range(self.control_args['lower_num_processes']))

        while active_workers:
            try:
                rank, memory = memory_queue.get(timeout=60) # Add a timeout to prevent infinite waiting

                if memory is None:
                    active_workers.remove(rank)
                else:
                    all_memories.append(memory)
                    print(f"Memory from worker {rank} received. Memory size: {len(memory.states)}")

                    self.action_timesteps += len(memory.states)
                    # Update lower level PPO every n times action has been taken
                    if self.action_timesteps % self.control_args['lower_update_freq'] == 0:
                        loss = self.lower_ppo.update(all_memories, agent_type='lower')

                        total_lower_reward = sum(sum(memory.rewards) for memory in all_memories)
                        avg_lower_reward = total_lower_reward / self.control_args['lower_num_processes'] # Average reward per process in this iteration
                        print(f"\nAverage Reward per process: {avg_lower_reward:.2f}\n")
                        
                        # clear memory to prevent memory growth (after the reward calculation)
                        for memory in all_memories:
                            memory.clear_memory()

                        # reset all memories
                        del all_memories #https://pytorch.org/docs/stable/multiprocessing.html
                        all_memories = []

                        # logging after update
                        if loss is not None:
                            if self.is_sweep: # Wandb for hyperparameter tuning
                                wandb.log({     "iteration": iteration,
                                                "lower_avg_reward": avg_lower_reward, # Set as maximize in the sweep config
                                                "lower_policy_loss": loss['policy_loss'],
                                                "lower_value_loss": loss['value_loss'], 
                                                "lower_entropy_loss": loss['entropy_loss'],
                                                "lower_total_loss": loss['total_loss'],
                                                "lower_current_lr": current_lr if self.control_args['lower_anneal_lr'] else self.control_args['lr'],
                                                "global_step": global_step          })
                                
                            else: # Tensorboard for regular training
                                total_updates = int(self.action_timesteps / self.control_args['lower_update_freq'])
                                self.writer.add_scalar('Lower/Average_Reward', avg_lower_reward, global_step)
                                self.writer.add_scalar('Lower/Total_Policy_Updates', total_updates, global_step)
                                self.writer.add_scalar('Lower/Policy_Loss', loss['policy_loss'], global_step)
                                self.writer.add_scalar('Lower/Value_Loss', loss['value_loss'], global_step)
                                self.writer.add_scalar('Lower/Entropy_Loss', loss['entropy_loss'], global_step)
                                self.writer.add_scalar('Lower/Total_Loss', loss['total_loss'], global_step)
                                self.writer.add_scalar('Lower/Current_LR', current_lr, global_step)
                                print(f"Logged lower agent data at step {global_step}")

                                # Save model every n times it has been updated (may not every iteration)
                                if self.control_args['save_freq'] > 0 and total_updates % self.control_args['save_freq'] == 0:
                                    torch.save(self.lower_ppo.policy.state_dict(), os.path.join(self.control_args['save_dir'], f'control_model_iteration_{iteration+1}.pth'))

                                # Save best model so far
                                if avg_lower_reward > self.best_reward_lower:
                                    torch.save(self.lower_ppo.policy.state_dict(), os.path.join(self.control_args['save_dir'], 'best_control_model.pth'))
                                    self.best_reward_lower = avg_lower_reward
                        
                        else: # For some reason..
                            print("Warning: loss is None")

            except queue.Empty:
                print("Timeout waiting for worker. Continuing...")
        
        # At the end of an iteration, wait for all processes to finish
        # The join() method is called on each process in the processes list. This ensures that the main program waits for all processes to complete before continuing.
        for p in processes:
            p.join()

        average_higher_reward = self._get_reward(iteration)

        iterative_torch_graph = self._convert_to_torch_geometric(self.iterative_networkx_graph)
        next_state = Data(x=iterative_torch_graph.x,
                           edge_index=iterative_torch_graph.edge_index,
                           edge_attr=iterative_torch_graph.edge_attr)
                      
        return next_state, average_higher_reward, done, info

    def _apply_action(self, proposals, iteration):
        """
        Every iteration, new proposals are added to networkx graph. Then that is converted to torch geometric (for state) and XML (for SUMO)
        Updates all three graph representations (networkx, torch, XML) based on the action.
        The proposals are expected to be a list of tuples (location, thickness) for each proposed crosswalk.
        
        Process:
        1. Denormalize proposed locations (thickness doesn't need denormalization)
        2. Add proposed crosswalks to networkx graph
        3. Update XML
        """

        # First make a copy
        self.iterative_networkx_graph = self.base_networkx_graph.copy()
        latest_horizontal_nodes_top_ped = self.horizontal_nodes_top_ped
        latest_horizontal_nodes_bottom_ped = self.horizontal_nodes_bottom_ped

        for i, (location, thickness) in enumerate(proposals):
            
            # 1. Denormalize the location (x-coordinate)
            denorm_location = self.normalizer_x['min'] + location * (self.normalizer_x['max'] - self.normalizer_x['min'])
            #print(f"\nLocation: {location} Denormalized location: {denorm_location}\n")

            # 2. Add to base networkx graph
            # Add new nodes in both sides in this intersection of type 'regular'.
            # Connect the new nodes to the existing nodes via edges with the given thickness.

            latest_horizontal_segment = self._get_horizontal_segment_ped(latest_horizontal_nodes_top_ped, latest_horizontal_nodes_bottom_ped, self.iterative_networkx_graph) # Start with set of nodes in base graph
            new_intersects = self._find_intersects_ped(denorm_location, latest_horizontal_segment, self.iterative_networkx_graph)
            #print(f"\nNew intersects: {new_intersects}\n")

            mid_node_details = {'top': {'y_cord': None, 'node_id': None}, 'bottom': {'y_cord': None, 'node_id': None}}
            # Now add an edge from from_node to the pos of the new node. As well as from the new node to to_node.
           
            for side in ['top', 'bottom']:
                 # Remove the old edge first.
                from_node, to_node = new_intersects[side]['edge'][0], new_intersects[side]['edge'][1]
                self.iterative_networkx_graph.remove_edge(from_node, to_node)

                # Add the new edge  
                end_node_pos = new_intersects[side]['intersection_pos']
                end_node_id = f"iter{iteration}_{i}_{side}"
                self.iterative_networkx_graph.add_node(end_node_id, pos=end_node_pos, type='regular')# type for this is regular

                self.iterative_networkx_graph.add_edge(from_node, end_node_id, width=2.0) # The width of these edges is default (Not from the proposal)
                self.iterative_networkx_graph.add_edge(end_node_id, to_node, width=2.0)

                # Modify the horizontal segment (add the new node)
                if side == 'top':
                    latest_horizontal_nodes_top_ped.append(end_node_id)
                else:
                    latest_horizontal_nodes_bottom_ped.append(end_node_id)

                mid_node_details[side]['y_cord'] = end_node_pos[1]
                mid_node_details[side]['node_id'] = end_node_id

            # Add the mid node and edges 
            mid_node_id = f"iter{iteration}_{i}_mid"
            mid_node_pos = (denorm_location, (mid_node_details['top']['y_cord'] + mid_node_details['bottom']['y_cord']) / 2)
            self.iterative_networkx_graph.add_node(mid_node_id, pos=mid_node_pos, type='middle')
            self.iterative_networkx_graph.add_edge(mid_node_details['top']['node_id'], mid_node_id, width=thickness) # Thickness is from sampled proposal
            self.iterative_networkx_graph.add_edge(mid_node_id, mid_node_details['bottom']['node_id'], width=thickness) # Thickness is from sampled proposal

        if self.design_args['save_graph_images']:
            save_graph_visualization(graph=self.iterative_networkx_graph, iteration=iteration)
            save_better_graph_visualization(graph=self.iterative_networkx_graph, iteration=iteration)

        # 3. Update XML
        self._update_xml_files(self.iterative_networkx_graph, iteration)
    
    def _find_segment_intersects_ped(self, segments, x_location):
        """
        Helper function to check intersection. 
        x_location is denormalized.
        """
        
        for start_x, (length, edge) in segments.items():
            end_x = start_x + length
            if start_x <= x_location < end_x:
                return {
                    'edge': edge,
                    'start_x': start_x,
                    'length_x': length
                }
                    
    def _find_intersects_ped(self, x_location, latest_horizontal_segment, latest_graph):
        """
        Find where a given x-coordinate intersects with the horizontal segments.
        Returns the edge IDs and positions where the intersection occurs.
        The graph is always changing as edges are added/removed.
        """
        intersections = {}

        for side in ['top', 'bottom']:
            intersections[side] = {}
            intersect = self._find_segment_intersects_ped(latest_horizontal_segment[side], x_location)

            from_node, to_node = intersect['edge'][0], intersect['edge'][1]
            
            # Extract node positions
            from_x, from_y = latest_graph.nodes[from_node]['pos']
            to_x, to_y = latest_graph.nodes[to_node]['pos']

            # Ensure from_x < to_x for consistency
            if from_x > to_x:
                from_x, to_x = to_x, from_x
                from_y, to_y = to_y, from_y

            # Compute how far along the segment x_location lies as a fraction
            x_diff = (x_location - from_x) / (to_x - from_x)
            # Now simply interpolate y
            y_location = from_y + x_diff * (to_y - from_y)

            intersections[side]['edge'] = intersect['edge']
            intersections[side]['intersection_pos'] = (x_location, y_location)

        return intersections

    def _get_horizontal_segment_ped(self, horizontal_nodes_top, horizontal_nodes_bottom, latest_graph, validation=False):
        """
        Get the entire horizontal segment of the corridor.
        """

        base_nodes_dict = {node[0]: node[1] for node in latest_graph.nodes(data=True)}
        edges = list(latest_graph.edges(data=True))

        horizontal_segment = {'top': {}, 'bottom': {}}

        # find edge ids horizontal_edges_top, horizontal_edges_bottom in edges
        for edge in edges:
            from_node, to_node = edge[0], edge[1]
            from_node_x, to_node_x = base_nodes_dict[from_node]['pos'][0], base_nodes_dict[to_node]['pos'][0]
            if from_node in horizontal_nodes_top and to_node in horizontal_nodes_top:
                smaller_x, larger_x = min(from_node_x, to_node_x), max(from_node_x, to_node_x)
                horizontal_segment['top'][smaller_x] = [larger_x - smaller_x, edge] #[2]['id']] # starting position, length, edge id
            elif from_node in horizontal_nodes_bottom and to_node in horizontal_nodes_bottom:
                smaller_x, larger_x = min(from_node_x, to_node_x), max(from_node_x, to_node_x)
                horizontal_segment['bottom'][smaller_x] = [larger_x - smaller_x, edge] #[2]['id']] # starting position, length, edge id

        # print(f"\nHorizontal top: {horizontal_segment['top']}\n")
        # print(f"\nHorizontal bottom: {horizontal_segment['bottom']}\n")

        # validation plot (to see if they make continuous horizontal segments)
        if validation:
            _, ax = plt.subplots()
            horizontal_segment_top = sorted(list(horizontal_segment['top'].keys()))
            horizontal_segment_bottom = sorted(list(horizontal_segment['bottom'].keys()))
            for start_pos in horizontal_segment_top:
                x_min, x_max = horizontal_segment_top[0], horizontal_segment_top[-1]
                length = horizontal_segment['top'][start_pos][0]
                ax.plot([start_pos, start_pos + length], [2, 2], 'r-')
                ax.plot(start_pos, 2, 'x')

                # plot the min and max x-coordinate values
                ax.text(x_min, 2, f'{x_min:.2f}', fontsize=12, verticalalignment='bottom')
                ax.text(x_max, 2, f'{x_max:.2f}', fontsize=12, verticalalignment='bottom')

            for start_pos in horizontal_segment_bottom:
                x_min, x_max = horizontal_segment_bottom[0], horizontal_segment_bottom[-1]
                length = horizontal_segment['bottom'][start_pos][0]
                ax.plot([start_pos, start_pos + length], [8, 8], 'b-')
                ax.plot(start_pos, 8, 'x')

                # plot the min and max x-coordinate values
                ax.text(x_min, 8, f'{x_min:.2f}', fontsize=12, verticalalignment='bottom')
                ax.text(x_max, 8, f'{x_max:.2f}', fontsize=12, verticalalignment='bottom')

            ax.set_ylim(-1, 11)
            ax.set_xlabel('X-coordinate')
            plt.savefig('./horizontal_segments.png')
            #plt.show()
        
        return horizontal_segment

    def _get_horizontal_segment_veh(self, horizontal_edges_top, horizontal_edges_bottom, latest_graph, validation=False):
        """
        Get the entire horizontal segment of the corridor.
        """
        pass 

    def _get_reward(self, iteration):
        """
        Design reward based on:
        - Pedestrians: how much time (on average) did it take for pedestrians to reach the nearest crosswalk

        """
        return 0


    def reset(self, start_from_base=False):
        """
        Reset the environment to its initial state.
        Option to start with the initial original set of crosswalks or start with an empty canvas.
        - Return state extracted from iterative torch graph
        """

        self.iterative_networkx_graph = self.base_networkx_graph.copy()

        if start_from_base:
            pass # Do nothing
        else: 
            pass
            # Add middle nodes and edges in the networkx graph. 
            # This middle node configuration will be slightly different from the middle nodes present in the original. 
            for cid, crosswalk_data in self.existing_crosswalks.items():
                # End nodes are already present in the networkx graph, add a connecting edge between them.
                bottom_pos, top_pos = crosswalk_data['pos'][0], crosswalk_data['pos'][-1]
                # create a new middle pos
                middle_x = (bottom_pos[0] + top_pos[0]) / 2
                middle_y = (bottom_pos[1] + top_pos[1]) / 2
                middle_pos = (middle_x, middle_y)

                # sanitize the id 
                cid = cid.replace(":", "") # Do not use a space
                middle_node_id = f"{cid}_mid"
                # Add the new middle node to the networkx graph
                self.iterative_networkx_graph.add_node(middle_node_id, pos=middle_pos, type='middle')

                # Add the connecting edge between the end nodes
                crossing_nodes = crosswalk_data['crossing_nodes']
                bottom_node, top_node = crossing_nodes[0], crossing_nodes[-1]
                self.iterative_networkx_graph.add_edge(bottom_node, middle_node_id, width=3.0)
                self.iterative_networkx_graph.add_edge(middle_node_id, top_node, width=3.0)

        if self.design_args['save_graph_images']:
            save_graph_visualization(graph=self.iterative_networkx_graph, iteration=0)
            save_better_graph_visualization(graph=self.iterative_networkx_graph, iteration=0)

        # Everytime the networkx graph is updated, the XML graph needs to be updated.
        # Make the added nodes/edges a crossing with traffic light in XML.
        self._update_xml_files(self.iterative_networkx_graph, 0)
        
        # Return state
        iterative_torch_graph = self._convert_to_torch_geometric(self.iterative_networkx_graph)
        state = Data(x=iterative_torch_graph.x,
                    edge_index=iterative_torch_graph.edge_index,
                    edge_attr=iterative_torch_graph.edge_attr)
        
        return state

    def close(self):
        """
        
        """
        pass
    
    def _cleanup_graph(self, graph, existing_crosswalks):
        """
        This step creates a base graph upon which additional edges are added/ removed during training.
        Multiple things happen (Requires some manual input):

        0. Primary cleanup:
            - In the original graph, at least 3 nodes exist to create a crosswalk.
            - To create a base graph, we need to remove these nodes in the middle. 
            - This clears existing crosswalks in the corridor. 

        1. Remove nodes and edges too far away from the corridor based on y values (Not used)
            - Clear the corridor of any nodes and edges that are irrelevant to the pedestrian walkway.
        
        2. Remove isolated and fringe nodes. (Not used: In the new approach, there are no fringe or isolated nodes.)
            - They could have existed because some vehicle roads allow pedestrians.
        """

        cleanup_graph = graph.copy()
        #print(f"\nBefore cleanup: {len(cleanup_graph.nodes())} nodes, {len(cleanup_graph.edges())} edges\n")

        # 0. primary cleanup 
        #print(f"Existing crosswalks: {existing_crosswalks}")
        middle_nodes = []
        for crosswalk_data in existing_crosswalks.values():
            nodes = crosswalk_data['crossing_nodes'] # There will always be more than 2 nodes.
            middle_nodes.extend(nodes[1:-1]) # Add all nodes except first and last
                
        #print(f"Removing middle nodes: {middle_nodes}")
        cleanup_graph.remove_nodes_from(middle_nodes)

        # 1.
        # remove everything with y-coordinates outside 10% and 90% range
        # y_coords = [data['pos'][1] for _, data in cleanup_graph.nodes(data=True)]
        # min_y, max_y = min(y_coords), max(y_coords)
        
        # y_range = max_y - min_y
        # y_lower = min_y + y_range * 0.1
        # y_upper = min_y + y_range * 0.9
        # nodes_to_remove_2 = [node for node, data in cleanup_graph.nodes(data=True)
        #                   if data['pos'][1] < y_lower or data['pos'][1] > y_upper]
                
        # cleanup_graph.remove_nodes_from(nodes_to_remove_2)
        # cleanup_graph.remove_edges_from(cleanup_graph.edges(nodes_to_remove_2))

        # 2.
        # fringe_nodes = ['9727779406','9740484031','cluster_9740411700_9740411702','9740155241','9740484518', '9740484521']
        # isolated_nodes = list(nx.isolates(cleanup_graph))
        # isolated_and_fringe_nodes = fringe_nodes + isolated_nodes
        # cleanup_graph.remove_nodes_from(isolated_and_fringe_nodes)
        #cleanup_graph.remove_edges_from(cleanup_graph.edges(isolated_and_fringe_nodes))  # TODO: When the isolated nodes are removed, are the edges automatically removed?
        #print(f"\nAfter cleanup: {len(cleanup_graph.nodes())} nodes, {len(cleanup_graph.edges())} edges\n")

        return cleanup_graph
    
    
    
    def _convert_to_torch_geometric(self, graph):
        """
        Converts the NetworkX graph to a PyTorch Geometric Data object.
        Normalizes the coordinates to lie between 0 and 1 and scales the width values proportionally.
        """
        # Create a mapping from node IDs to indices
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(graph.nodes())}

        # Extract node features (x, y coordinates)
        node_features = []
        for node_id in graph.nodes():
            data = graph.nodes[node_id]
            node_features.append([data['pos'][0], data['pos'][1]])

        x = torch.tensor(node_features, dtype=torch.float)
        x = self._normalize_features(x)
        
        # Extract edge indices and attributes
        edge_index = []
        edge_attr = []
        for source_id, target_id, edge_data in graph.edges(data=True):
            source = node_id_to_index[source_id]
            target = node_id_to_index[target_id]
            edge_index.append([source, target])
            edge_index.append([target, source]) # Add reverse edge (for undirected graph)

            # Get source node's x coordinate and add it to edge attribute.
            # TODO: come back to this.
            source_x = (graph.nodes[source_id]['pos'][0] - self.normalizer_x['min']) / (self.normalizer_x['max'] - self.normalizer_x['min'])
            edge_attr.append([edge_data['width'], source_x])  # Add source x-coordinate alongside width
            edge_attr.append([edge_data['width'], source_x])  # For the reverse edge as well.
        
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

        normalized_x = (x_coords - self.normalizer_x['min']) / (self.normalizer_x['max'] - self.normalizer_x['min'])
        normalized_y = (y_coords - self.normalizer_y['min']) / (self.normalizer_y['max'] - self.normalizer_y['min'])

        return torch.stack([normalized_x, normalized_y], dim=1)
    
    def _update_xml_files(self, networkx_graph, iteration):
        """
        Update the XML component files to reflect the current state of the networkx graph. 
        For base, use the "original" XML component files. For other iterations, use the "base" XML component files as a foundation and add/ remove elements.
        Iterative component files are saved in component_SUMO_files directory.
        Iterative net files are saved in network_iterations directory.

        Networkx graph will already have:
          - End nodes with position values that come from the proposal.
          - Middle nodes to create traffic lights. Every proposal will have nodes with id _mid. 
                - The _mid nodes need to be connected to vehicle edges on either side as well. 
                - For every _mid node, a TL logic needs to be added to the traffic_light XML file.

        For iteration = base, the end nodes are already present in the XML component files.
        For other iterations, the end nodes will require reconstruction.

        For the nodes and edges related to pedestrian network:
            Remove them from the XML component files: If they don't exist in networkx graph.
            Add them to the XML component files: If they exist in networkx graph but don't exist in component XML.

        Node attributes: 
            - <node id=" " x=" " y=" " type=" " />
            - For the middle nodes, type will "traffic_light" and an attribute tl =" " with the value same as id.
            - For the end nodes, type will be "dead_end"
        Edge attributes: 
            - From the middle node to end nodes of type "highway.footway"
            - From middle node to vehicle nodes of type "highway.tertiary"
            - Both of these are needed because the traffic light is coordinating vehicles and pedestrians in the crosswalk.
            - <edge id=" " from=" " to=" " priority="1" type="highway.footway" numLanes="1" speed="2.78" shape=" " spreadType="center" width="2.0" allow="pedestrian"> 
            - shape seems difficult to get right.
            - create a nested lane element: <lane index="0" allow="pedestrian" width=" " speed=" ">
            - create a nested param element: <param key="origId" value=" "/>
            - end with </lane></edge>
        """

        prefix = "original" if iteration == 'base' else f"iteration_base" # Every iteration will have the same base XML files.
        node_file = f'{self.component_dir}/{prefix}.nod.xml'
        edge_file = f'{self.component_dir}/{prefix}.edg.xml'
        connection_file = f'{self.component_dir}/{prefix}.con.xml'
        type_file = f'{self.component_dir}/{prefix}.typ.xml'
        traffic_light_file = f'{self.component_dir}/{prefix}.tll.xml'

        # Parse the XML files
        node_tree = ET.parse(node_file)
        edge_tree = ET.parse(edge_file)
        connection_tree = ET.parse(connection_file)
        type_tree = ET.parse(type_file)
        traffic_light_tree = ET.parse(traffic_light_file)

        node_root = node_tree.getroot()
        edge_root = edge_tree.getroot()
        connection_root = connection_tree.getroot()
        type_root = type_tree.getroot() 
        traffic_light_root = traffic_light_tree.getroot()

        # Find all the nodes and edges in the XML component files ()
        nodes_in_xml = { node.get('id'): node for node in node_root.findall('node') } # save the node element itself.
        edges_in_xml = { (e.get('from'), e.get('to')): e for e in edge_root.findall('edge') } # save the edge element itself.

        # Find pedestrian nodes and edges in the XML component .edg file. 
        pedestrian_edges_in_xml = {}
        pedestrian_nodes_in_xml = set()
        for (f, t), e in edges_in_xml.items():
            e_type = e.get('type')
            allow = e.get('allow', '')
            if e_type in ['highway.footway', 'highway.steps'] and 'pedestrian' in allow:
                pedestrian_edges_in_xml[(f, t)] = e
                pedestrian_nodes_in_xml.update([f, t]) # From node id alone we cant differentiate between vehicle and pedestrian nodes.

        # Extract pedestrian nodes and edges from networkx_graph
        pedestrian_nodes_in_graph = set(networkx_graph.nodes())
        pedestrian_edges_in_graph = set(networkx_graph.edges())
        
        print(f"Pedestrian nodes in XML: {pedestrian_nodes_in_xml}\n")
        print(f"Pedestrian nodes in graph: {pedestrian_nodes_in_graph}\n")
        print(f"Pedestrian edges in XML: {list(pedestrian_edges_in_xml.keys())}\n")
        print(f"Pedestrian edges in graph: {pedestrian_edges_in_graph}\n")
        
        # Remove pedestrian nodes that are in XML component file but not in networkx graph.
        potential_nodes_to_remove = pedestrian_nodes_in_xml - pedestrian_nodes_in_graph
        print(f"Potential Nodes to remove: Total: {len(potential_nodes_to_remove)},\n {potential_nodes_to_remove}\n")
        
        # Some edges may still access the nodes that are removed above.
        # Find the edges that still access the nodes that are removed above.
        edges_in_xml_that_access_removed_nodes = {}
        for (f, t) in edges_in_xml:
            if f in potential_nodes_to_remove or t in potential_nodes_to_remove:
                edges_in_xml_that_access_removed_nodes[(f, t)] = edges_in_xml[(f, t)] # These can be vehicle edges as well.

        print(f"Edges in XML that still access the potential removed nodes: Total: {len(edges_in_xml_that_access_removed_nodes)}")
        for (f, t), e in edges_in_xml_that_access_removed_nodes.items():
            print(f"Edge: {f} -> {t}")
            print(f"Edge attributes: {e.attrib}\n")

        # In the potential nodes selected for removal, some of them are vehicle edges (The old TL was removed) (highway.tertiary and disallowed pedestrian).
        vehicle_edges_linked_to_potential_removal_nodes = {}
        for (f, t), e in edges_in_xml_that_access_removed_nodes.items():
            e_type = e.get('type')
            disallow = e.get('disallow', '')
            if e_type == 'highway.tertiary' and 'pedestrian' in disallow:
                vehicle_edges_linked_to_potential_removal_nodes[(f, t)] = e
        print(f"Vehicle edges linked to potential removal nodes: Total: {len(vehicle_edges_linked_to_potential_removal_nodes)},\n {vehicle_edges_linked_to_potential_removal_nodes}\n")

        # Each removed node that is a part of the vehicle edges will have four vehicle edges (two for each direction).
        # Option 1: Merge into a larger edge by combining the shapes. This may be harmful because the edge itself could be referenced by other other parts of the network.
        # Option 2: keep these vehicle nodes and edges as they are. Only remove the pedestrian edges and the TL associated with these nodes.
        # Others are pedestrian nodes and edges, they can be removed.
        
        # Get all nodes that appear in vehicle edges
        nodes_in_vehicle_edges = set()
        for f, t in vehicle_edges_linked_to_potential_removal_nodes.keys():
            nodes_in_vehicle_edges.add(f)
            nodes_in_vehicle_edges.add(t)

        print(f"Potential nodes to be removed: {potential_nodes_to_remove}\n")
        print(f"Nodes in vehicle edges: {nodes_in_vehicle_edges}\n")
        
        # Remove nodes that appear in vehicle edges from potential nodes to remove
        pedestrian_nodes_to_remove = potential_nodes_to_remove - nodes_in_vehicle_edges
        print(f"Actual pedestrian nodes to remove: Total: {len(pedestrian_nodes_to_remove)},\n {pedestrian_nodes_to_remove}\n")

        # Remove the pedestrian edges that are not present in the vehicle edges.
        pedestrian_edges_to_remove = {}
        for (f, t), e in edges_in_xml_that_access_removed_nodes.items():
            if (f, t) not in vehicle_edges_linked_to_potential_removal_nodes:
                pedestrian_edges_to_remove[(f, t)] = e
        print(f"Actual pedestrian edges to remove: Total: {len(pedestrian_edges_to_remove)}, \n {pedestrian_edges_to_remove}\n")

        # Removing selected nodes and edges
        for node_id in pedestrian_nodes_to_remove:
            if node_id in nodes_in_xml:
                node_root.remove(nodes_in_xml[node_id]) # remove from nod component file
                del nodes_in_xml[node_id] # remove from dictionary

        for (f, t) in pedestrian_edges_to_remove:
            if (f, t) in edges_in_xml:
                edge_root.remove(edges_in_xml[(f, t)]) # remove from edg component file
                del edges_in_xml[(f, t)] # remove from dictionary

        # Remove the associated TLs (This is safe I think.)
        # Remove all TLs and connections except the default one.
        default_tl = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # By default in base, there will the one TL at the left intersection. present.
        tls_to_remove = []
        for tl in traffic_light_root.findall('tlLogic'):
            if tl.get('id') not in default_tl:
                tls_to_remove.append(tl)
        for tl in tls_to_remove:
            traffic_light_root.remove(tl)

        # Since the vehicle nodes are not removed, perhaps the connections dont need to be removed (because they connect the vehicle edges). However, removing them for now.
        # Perhaps each connection here needs to be converted to an edge? No.
        connections_to_remove = []
        for conn in traffic_light_root.findall('connection'):
            if conn.get('tl') not in default_tl:
                connections_to_remove.append(conn)
        for conn in connections_to_remove:
            traffic_light_root.remove(conn)

        # Node additions at the end nodes (in iterations other than base) and middle nodes.
        # Find the pedestrian nodes to add (present in networkx graph but not in XML component file). 
        # For regular nodes: <node id=" " x=" " y=" " />
        # For all the nodes in networkx graph with type "middle" in the XML component file: add a node attribute type with value "traffic_light" and tl with value "node_id" and add a TL logic and a connection.
        node_ids_to_add = pedestrian_nodes_in_graph - set(nodes_in_xml.keys()) # In the iteration base, there will be no nodes to add.
        print(f"\nNodes to add: {node_ids_to_add}")

        for nid in node_ids_to_add:
            node_data = networkx_graph.nodes[nid]
            x, y = node_data['pos']
            n_type = node_data.get('type', 'regular')
            attribs = {'id': nid, 'x': str(x), 'y': str(y)}

            if n_type == 'regular':
                attribs['type'] = 'dead_end'
            elif n_type == 'middle':
                attribs['type'] = 'traffic_light'
                attribs['tl'] = nid

            new_node = ET.Element('node', attribs)
            node_root.append(new_node)
            nodes_in_xml[nid] = new_node

            
        
        # Edge additions
        # Find the edges to add (present in networkx graph but not in XML component file).
        # For each edge, add a new edge element with the same attributes.
        edges_to_add = set(networkx_graph.edges()) - set(edges_in_xml.keys()) # These are all pedestrian edges.
        edges_to_add = list(edges_to_add)
        print(f"\nEdges to add: Total: {len(edges_to_add)},\n {edges_to_add}\n")

        # The edge could be a Regular node to Regular node or Regular node to middle node (crosswalk).
        for (f, t) in edges_to_add:

            # Do Regular to Regular and Regular to Middle need some different treatment?
            edge_data = networkx_graph.get_edge_data(f, t)
            edge_id = edge_data.get('id', f'edge_{f}_{t}') # Get it from the networkx graph.
            width = edge_data.get('width', 2.0)
            edge_attribs = {
                'id': edge_id,
                'from': f,
                'to': t,
                'name': 'Iterative addition',
                'priority': '1',
                'type': 'highway.footway',
                'numLanes': '1',
                'speed': '2.78', # default
                'spreadType': 'center',
                'width': str(width),
                'allow': 'pedestrian'
            }

            # positions of f and t nodes
            f_data = networkx_graph.nodes[f]
            t_data = networkx_graph.nodes[t]
            f_x, f_y = round(f_data['pos'][0], 2), round(f_data['pos'][1], 2)
            t_x, t_y = round(t_data['pos'][0], 2), round(t_data['pos'][1], 2)
            shape = f'{f_x},{f_y} {t_x},{t_y}'
            #print(f"Shape: {shape}")

            edge_element = ET.Element('edge', edge_attribs)
            edge_element.text = "\n\t\t" 

            lane_element = ET.SubElement(
                edge_element,
                'lane', 
                index='0', 
                allow='pedestrian',
                width=str(width),
                speed='2.78', 
                shape=shape
            )

            lane_element.text = "\n\t\t\t" 
            param_element = ET.SubElement(lane_element, 'param', key='origId', value=edge_id)
            param_element.tail = "\n\t\t" 
            lane_element.tail = "\n\t"
            edge_element.tail = "\n\t"

            edge_root.append(edge_element)

        # TODO: Vehicle connections.
        # For all the mid nodes, find and add all vehicle connections: that connect the TL to the existing vehicle edges. 
        # The vehicle edges that are already present in either sides of the newly added TL (middle node). 
        middle_nodes_to_add = [nid for nid in node_ids_to_add if networkx_graph.nodes[nid].get('type') == 'middle']
        removed_vehicle_edges = [] # ids of the vehicle edges that are removed.
        # Whenever a middle node falls on a vehicle edge:
        # Split the edge into two edges and add a new connection (both to TLL and Con files). Use linkIndex 0 for connecting -ve direction and linkIndex 1 for connecting +ve direction.
        # The new edges should inherit certain properties of the original edge to preserve the shape etc.
        # Delete the original edge (add the id to the edge removed_vehicle_edges list).
        xml_horizontal_segments_base = {}


        # TODO: For all the node ids with _middle (add a crossing tag in the connection file) with vehicle edges on either side as edges attributes.
        # First remove all the crossing tags except default one.
        for crossing in connection_root.findall('crossing'):
            if crossing.get('node') not in default_tl:
                connection_root.remove(crossing)
        # The width here needs to come from the model. 
        
        # TODO: Might need to update the y_coordinate of the mid nodes to align with the vehicle edges/ nodes? That would result in a change to networkx graph.

        # for node_id in nodes_to_add:
        #     node_data = networkx_graph.nodes[node_id]
        #     if node_data.get('type') == 'middle':
        #         from_node =  # TODO: Figure these out.. are these vehicle nodes?
        #         to_node =  # TODO: Figure these out.. are these vehicle nodes?
        #         conn_attribs = {'from': from_node, 'to': to_node, 'tl': node_id, 'fromLane': '0', 'toLane': '0'} # Since inside the corridor, there is only one lane.
        #         connection_element = ET.Element('connection', conn_attribs)
        #         traffic_light_root.append(connection_element)

        # TL logic additions
        for nid in middle_nodes_to_add:
            node_data = networkx_graph.nodes[nid]
            tlLogic_element = ET.Element('tlLogic', id=nid, type='static', programID='0', offset='0')
            tlLogic_element.text = "\n\t\t" # Inside <tlLogic>: phases start at two tabs

            # Create phases with proper indentation
            phase1 = ET.SubElement(tlLogic_element, 'phase', duration='77', state='GGr')
            phase1.tail = "\n\t\t"
            phase2 = ET.SubElement(tlLogic_element, 'phase', duration='3', state='yyr') 
            phase2.tail = "\n\t\t"
            phase3 = ET.SubElement(tlLogic_element, 'phase', duration='5', state='rrG')
            phase3.tail = "\n\t\t"
            phase4 = ET.SubElement(tlLogic_element, 'phase', duration='5', state='rrr')
            phase4.tail = "\n\t"

            tlLogic_element.tail = "\n\t"
            traffic_light_root.append(tlLogic_element)
                
        # Respective changes to the connections file.
        # All the connections present in the TLL file should also be present in the connections file. But the connection file will have more of them.
        # In iteration base, there will be a bunch of connections to remove from original file (remove connections with the same from and to edges).

        all_conn_file_connections = [(conn.get('from'), conn.get('to')) for conn in connection_root.findall('connection')]
        print(f"connection Before removal: Total: {len(all_conn_file_connections)},\n {all_conn_file_connections}\n")
        
        # Look at the same from and to edges in the connections file and remove them.
        connections_to_remove_list = [(conn.get('from'), conn.get('to')) for conn in connections_to_remove]
        #print(f"connections_to_remove_list: Total: {len(connections_to_remove_list)},\n {connections_to_remove_list}\n")

        to_remove = []
        for conn in connection_root.findall('connection'):
            from_edge = conn.get('from')
            to_edge = conn.get('to')
            if (from_edge, to_edge) in connections_to_remove_list:
                to_remove.append(conn)

        # Remove all identified connections from connection file
        for conn in to_remove:
            connection_root.remove(conn)


        # Additional stuff related to edge removals.
        # If the edge (pedestrian and vehicle) is removed, then the connections to and from that edge should also be removed.
        pedestrian_edges_to_remove_connections = []
        for (f,t), edge in pedestrian_edges_to_remove.items():
            pedestrian_edges_to_remove_connections.append(edge.get('id'))

        print(f"pedestrian_edges_to_remove_connections: Total: {len(pedestrian_edges_to_remove_connections)},\n {pedestrian_edges_to_remove_connections}\n")

        for conn in connection_root.findall('connection'):
            if conn.get('from') in pedestrian_edges_to_remove_connections or conn.get('to') in pedestrian_edges_to_remove_connections:
                connection_root.remove(conn)

        all_conn_file_connections = [(conn.get('from'), conn.get('to')) for conn in connection_root.findall('connection')]
        print(f"connection After removal: Total: {len(all_conn_file_connections)},\n {all_conn_file_connections}\n")


        
        
        iteration_prefix = f'{self.component_dir}/iteration_{iteration}'
        node_tree.write(f'{iteration_prefix}.nod.xml', encoding='utf-8', xml_declaration=True)
        edge_tree.write(f'{iteration_prefix}.edg.xml', encoding='utf-8', xml_declaration=True)
        connection_tree.write(f'{iteration_prefix}.con.xml', encoding='utf-8', xml_declaration=True)
        type_tree.write(f'{iteration_prefix}.typ.xml', encoding='utf-8', xml_declaration=True)
        traffic_light_tree.write(f'{iteration_prefix}.tll.xml', encoding='utf-8', xml_declaration=True)

        # Generate the final net file using netconvert
        output_file = f'{self.network_dir}/network_iteration_{iteration}.net.xml'
        command = f"netconvert --node-files={iteration_prefix}.nod.xml --edge-files={iteration_prefix}.edg.xml --connection-files={iteration_prefix}.con.xml --type-files={iteration_prefix}.typ.xml --tllogic-files={iteration_prefix}.tll.xml --output-file={output_file}"

        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            if result.stderr:
                print(f"Warnings/Errors from netconvert: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error running netconvert: {e}")
            print("Error output:", e.stderr)

        # terminate the program here
        #os._exit(0)

    def _initialize_normalizers(self, graph):
        """
        Initialize normalizers based on the graph coordinates
        """
        # Extract all x and y coordinates from the graph
        coords = np.array([data['pos'] for _, data in graph.nodes(data=True)])
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        self.normalizer_x = {'min': float(np.min(x_coords)), 'max': float(np.max(x_coords))}
        self.normalizer_y = {'min': float(np.min(y_coords)), 'max': float(np.max(y_coords))}
