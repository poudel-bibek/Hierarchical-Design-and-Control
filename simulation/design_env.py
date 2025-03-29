import wandb
import subprocess
import gymnasium as gym
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as mp
import xml.etree.ElementTree as ET
from gymnasium import spaces
from torch_geometric.data import Data

from ppo.ppo import PPO
from ppo.ppo_utils import Memory
from .env_utils import *
from .sim_setup import CONTROLLED_CROSSWALKS_DICT, return_horizontal_nodes
from .worker import parallel_train_worker
from utils import save_policy, get_averages

class DesignEnv(gym.Env):
    """
    Higher level agent.
    - Modifies the net file based on design action.
    - No need to connect to or close this environment. 
    - Internally calls the parallel workers of the control environment.

    Improvements:
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
        self.design_args = design_args
        self.control_args = control_args
        self.lower_ppo_args = lower_ppo_args
        self.is_sweep = is_sweep
        self.is_eval = is_eval
        self.max_proposals = self.design_args['max_proposals']
        self.component_dir = self.design_args['component_dir']
        self.network_dir = self.design_args['network_dir']
        clear_folders(self.component_dir, self.network_dir) # Do not change the position of this.
        
        # Generate the 5 different component XML files (node, edge, connection, type, tllogic) from the net file.
        self._create_component_xml_files(self.design_args['original_net_file'])

        # Extract networkx graph from the component files. (Also update locations of nodes in existing_crosswalks)
        pedestrian_networkx_graph  = self._extract_networkx_graph() 
        self.horizontal_nodes_top_ped, self.horizontal_nodes_bottom_ped = return_horizontal_nodes()

        # Initialize normalizers
        self._initialize_normalizers(pedestrian_networkx_graph)
        self.min_thickness = self.design_args['min_thickness']
        self.max_thickness = self.design_args['max_thickness']

        self.existing_crosswalks = self._get_existing_crosswalks(pedestrian_networkx_graph) # Make use of the networkx graph to add locations
        
        # Cleanup the pedestrian walkway graph (i.e, remove isolated, fringe, existing crosswalks) to create a base canvas
        self.base_networkx_graph = self._cleanup_graph(pedestrian_networkx_graph, self.existing_crosswalks)
        
        self.horizontal_edges_veh_original_data = self._get_original_veh_edge_config()
        
        # Relevant to the intersection
        self.extreme_edge_dict = {'leftmost': {'old': "16666012#2", 'new': None},
                             'rightmost': {'old': "16666012#17", 'new': None}}
                             
        self._update_xml_files(self.base_networkx_graph, 'base') # Create base XML files from latest networkx graph

        if self.design_args['save_graph_images']:
            save_graph_visualization(graph=pedestrian_networkx_graph, iteration='original')
            save_graph_visualization(graph=self.base_networkx_graph, iteration='base')
            save_better_graph_visualization(graph=pedestrian_networkx_graph, iteration='original')
            save_better_graph_visualization(graph=self.base_networkx_graph, iteration='base')

        # Lower level agent
        self.lower_ppo = PPO(**self.lower_ppo_args)

        self.global_step = 0
        self.action_timesteps = 0 # keep track of how many times action has been taken by all lower level workers
        self.lower_update_count = 0

        self.writer = self.control_args['writer']
        self.best_reward_lower = float('-inf')
        self.total_updates_lower = None

        # Bugfix: Removing unpicklable object (writer) from control_args
        self.control_args_worker = {k: v for k, v in self.control_args.items() if k != 'writer'}
        self.model_init_params_worker = {'model_dim': self.lower_ppo_args['model_dim'],
                                     'action_dim': self.lower_ppo_args['action_dim'],
                                     'kwargs': self.lower_ppo_args['model_kwargs']}

    @property
    def action_space(self):
        """
        Both location and thickness are generated in the range [0, 1], then clamped to [0.01, 0.99], then denormalized.
        """
        return spaces.Dict({
            'num_proposals': spaces.Discrete(self.max_proposals),  # 0 to max_proposals
            'proposals': spaces.Box(
                low=np.array([[0, 0]] * self.max_proposals),
                high=np.array([[1, 1]] * self.max_proposals),
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
        output_dir = "./simulation/components"
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

    def step(self, 
             padded_proposals, 
             num_proposals, 
             iteration, 
             SEED, 
             shared_state_normalizer,
             shared_reward_normalizer,
             eval_args,
             is_sweep):
        
        """
        Every step in the design environment involves:
        - Updating the network xml file based on the design action.
        - A number of parallel workers (that utilize the new network file) to each carry out one episode in the control environment.
        """

        # print(f"\nHigher level action received: {action}\n")
        # Convert tensor action to proposals
        padded_proposals = padded_proposals.cpu().numpy()  # Convert to numpy array if it's not already
        proposals = padded_proposals[0][:num_proposals]  # Only consider the actual proposals
        print(f"\nProposals: {proposals}")

        # Apply the action to output the latest SUMO network file as well as modify the iterative_torch_graph.
        self._apply_action(proposals, iteration)

        done = False
        info = {}
        
        lower_old_policy = self.lower_ppo.policy_old.to(self.lower_ppo_args['device'])
        lower_old_policy.share_memory() # The same policy is shared among all workers.
        lower_old_policy.eval() # So that dropout, batchnorm, layer norm etc. are not used during inference

        lower_queue = mp.Queue()
        lower_processes = []
        active_lower_workers = []
        for rank in range(self.control_args['lower_num_processes']):
            worker_seed = SEED + iteration * 1000 + rank
            p = mp.Process(
                target=parallel_train_worker,
                args=(
                    rank,
                    lower_old_policy,
                    self.control_args_worker,
                    lower_queue,
                    worker_seed,
                    num_proposals,
                    shared_state_normalizer,
                    shared_reward_normalizer,
                    self.extreme_edge_dict,
                    self.lower_ppo_args['device'],
                    iteration)
                )
            p.start()
            lower_processes.append(p)
            active_lower_workers.append(rank)
             
        all_memories = Memory()
        while active_lower_workers:
            print(f"Active workers: {active_lower_workers}")
            rank, memory = lower_queue.get(timeout=30)

            if memory is None:
                print(f"Worker {rank} None received")
                active_lower_workers.remove(rank)
            else:
                current_action_timesteps = len(memory.states)
                print(f"Memory from worker {rank} received. Memory size: {current_action_timesteps}")
                all_memories.actions.extend(torch.from_numpy(np.asarray(memory.actions)))
                all_memories.states.extend(torch.from_numpy(np.asarray(memory.states)))
                all_memories.values.extend(memory.values)
                all_memories.logprobs.extend(memory.logprobs)
                all_memories.rewards.extend(memory.rewards)
                all_memories.is_terminals.extend(memory.is_terminals)

                self.action_timesteps += current_action_timesteps
                self.global_step += current_action_timesteps * self.control_args['lower_action_duration']
                print(f"Action timesteps: {self.action_timesteps}, global step: {self.global_step}")
                del memory #https://pytorch.org/docs/stable/multiprocessing.html

                # Update PPO every n times (or close to n) action has been taken 
                if self.action_timesteps >= self.control_args['lower_update_freq']:
                    print(f"Updating PPO with {len(all_memories.actions)} memories") 

                    self.lower_update_count += 1

                    # Anneal after every update
                    if self.control_args['lower_anneal_lr']:
                        current_lr_lower = self.lower_ppo.update_learning_rate(iteration, self.total_updates_lower)

                    avg_reward = sum(all_memories.rewards) / len(all_memories.rewards)
                    print(f"\nAverage Reward (across all memories): {avg_reward}\n")

                    loss = self.lower_ppo.update(all_memories)

                    # Reset all memories
                    del all_memories
                    all_memories = Memory() 
                    self.action_timesteps = 0
                    print(f"Size of all memories after update: {len(all_memories.actions)}")

                    # Save (and evaluate the latest policy) every save_freq updates
                    if self.lower_update_count % self.control_args['lower_save_freq'] == 0:
                        latest_policy_path = os.path.join(self.control_args['save_dir'], f'lower_policy_at_step_{self.global_step}.pth')
                        save_policy(self.lower_ppo.policy, shared_state_normalizer, latest_policy_path)

                        # Evaluate the latest policy
                        print(f"Evaluating policy: {latest_policy_path} at step {self.global_step}")
                        eval_json = eval(self.control_args_worker, self.lower_ppo_args, eval_args, policy_path=latest_policy_path, tl= False) # which policy to evaluate?
                        _, eval_veh_avg_wait, eval_ped_avg_wait, _, _ = get_averages(eval_json)
                        eval_veh_avg_wait = np.mean(eval_veh_avg_wait)
                        eval_ped_avg_wait = np.mean(eval_ped_avg_wait)
                        avg_eval = ((eval_veh_avg_wait + eval_ped_avg_wait) / 2)
                        print(f"Eval veh avg wait: {eval_veh_avg_wait}, eval ped avg wait: {eval_ped_avg_wait}, avg eval: {avg_eval}")

                    # Save best policies 
                    if avg_reward > best_reward:
                        save_policy(self.lower_ppo.policy, shared_state_normalizer, os.path.join(self.control_args['save_dir'], 'best_reward_policy.pth'))
                        best_reward = avg_reward
                    if loss['total_loss'] < best_loss:
                        save_policy(self.lower_ppo.policy, shared_state_normalizer, os.path.join(self.control_args['save_dir'], 'best_loss_policy.pth'))
                        best_loss = loss['total_loss']
                    if avg_eval < best_eval:
                        save_policy(self.lower_ppo.policy, shared_state_normalizer, os.path.join(self.control_args['save_dir'], 'best_eval_policy.pth'))
                        best_eval = avg_eval

                    # logging
                    if is_sweep: # Wandb for hyperparameter tuning
                        wandb.log({ "iteration": iteration,
                                        "avg_reward": avg_reward, # Set as maximize in the sweep config
                                        "update_count": self.lower_update_count,
                                        "policy_loss": loss['policy_loss'],
                                        "value_loss": loss['value_loss'], 
                                        "entropy_loss": loss['entropy_loss'],
                                        "total_loss": loss['total_loss'],
                                        "current_lr": current_lr_lower if self.control_args['lower_anneal_lr'] else self.lower_ppo_args['lr'],
                                        "approx_kl": loss['approx_kl'],
                                        "eval_veh_avg_wait": eval_veh_avg_wait,
                                        "eval_ped_avg_wait": eval_ped_avg_wait,
                                        "avg_eval": avg_eval,
                                        "global_step": self.global_step })
                        
                    else: # Tensorboard for regular training
                        self.writer.add_scalar('Lower/Average_Reward', avg_reward, self.global_step)
                        self.writer.add_scalar('Lower/Total_Policy_Updates', self.lower_update_count, self.global_step)
                        self.writer.add_scalar('Lower/Policy_Loss', loss['policy_loss'], self.global_step)
                        self.writer.add_scalar('Lower/Value_Loss', loss['value_loss'], self.global_step)
                        self.writer.add_scalar('Lower/Entropy_Loss', loss['entropy_loss'], self.global_step)
                        self.writer.add_scalar('Lower/Total_Loss', loss['total_loss'], self.global_step)
                        self.writer.add_scalar('Lower/Current_LR', current_lr_lower if self.control_args['lower_anneal_lr'] else self.lower_ppo_args['lr'], self.global_step)
                        self.writer.add_scalar('Lower/Approx_KL', loss['approx_kl'], self.global_step)
                        self.writer.add_scalar('Evaluation/Veh_Avg_Wait', eval_veh_avg_wait, self.global_step)
                        self.writer.add_scalar('Evaluation/Ped_Avg_Wait', eval_ped_avg_wait, self.global_step)
                        self.writer.add_scalar('Evaluation/Avg_Eval', avg_eval, self.global_step)
                        print(f"Logged lower agent data at step {self.global_step}")
                        
                        
         # Clean up. The join() method ensures that the main program waits for all processes to complete before continuing.
        for p in lower_processes:
            p.join() 
        print(f"All processes joined\n\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del lower_queue

        # Higher level agent's reward can only be obtained after the lower level workers have finished
        # It is also averaged across the various lower level workers.
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
            location = round(location.item(), 2) # round to prevent going out of bounds.
            thickness = round(thickness.item(), 2)

            # 1. Denormalize the location (x-coordinate) and thickness
            denorm_location = self.normalizer_x['min'] + location * (self.normalizer_x['max'] - self.normalizer_x['min'])
            print(f"\nLocation: {location} Denormalized location: {denorm_location}")

            denorm_thickness = self.min_thickness + thickness * (self.max_thickness - self.min_thickness)
            print(f"Thickness: {thickness} Denormalized thickness: {denorm_thickness}\n")

            # 2. Add to base networkx graph
            # Add new nodes in both sides in this intersection of type 'regular'.
            # Connect the new nodes to the existing nodes via edges with the given thickness.

            latest_horizontal_segment = self._get_horizontal_segment_ped(latest_horizontal_nodes_top_ped, latest_horizontal_nodes_bottom_ped, self.iterative_networkx_graph) # Start with set of nodes in base graph
            # print(f"\nLatest horizontal segment (Top): {latest_horizontal_segment['top']}\n")
            # print(f"\nLatest horizontal segment (Bottom): {latest_horizontal_segment['bottom']}\n")
            new_intersects = self._find_intersects_ped(denorm_location, latest_horizontal_segment, self.iterative_networkx_graph)
            print(f"\nNew intersects: {new_intersects}\n")

            mid_node_details = {'top': {'y_cord': None, 'node_id': None}, 'bottom': {'y_cord': None, 'node_id': None}}
            # Now add an edge from from_node to the pos of the new node. As well as from the new node to to_node.
           
            for side in ['top', 'bottom']:
                 # Remove the old edge first.
                from_node, to_node = new_intersects[side]['edge'][0], new_intersects[side]['edge'][1]
                self.iterative_networkx_graph.remove_edge(from_node, to_node)

                # Add the new edge  
                end_node_pos = new_intersects[side]['intersection_pos']
                end_node_id = f"iter{iteration}_{i}_{side}"
                self.iterative_networkx_graph.add_node(end_node_id, pos=end_node_pos, type='regular', width=-1) # type for this is regular (width specified for completeness as -1: Not used)
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
            
            # Obtain the y_coordinate of the middle node. Based on adjacent vehicle edges. Use interpolation to find the y coordinate.
            # To ensure that the y coordinates of the graph and the net file are the same. This has to be done here. 
            # previous method, midpoint
            # mid_node_pos = (denorm_location, (mid_node_details['top']['y_cord'] + mid_node_details['bottom']['y_cord']) / 2)

            # new method, interpolation. Always using the original vehicle edge list (not updated with split of split).
            mid_node_pos = (denorm_location, interpolate_y_coordinate(denorm_location, self.horizontal_edges_veh_original_data))

            self.iterative_networkx_graph.add_node(mid_node_id, pos=mid_node_pos, type='middle', width=denorm_thickness) 
            self.iterative_networkx_graph.add_edge(mid_node_details['top']['node_id'], mid_node_id, width=denorm_thickness) 
            self.iterative_networkx_graph.add_edge(mid_node_id, mid_node_details['bottom']['node_id'], width=denorm_thickness) 

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
        Find where a given x-coordinate intersects with the horizontal pedestriansegments.
        Returns the edge IDs and positions where the intersection occurs.
        The graph is always changing as edges are added/removed.
        """
        intersections = {}

        for side in ['top', 'bottom']:
            intersections[side] = {}
            intersect = self._find_segment_intersects_ped(latest_horizontal_segment[side], x_location)
            print(f"\nLatest horizontal segment: {latest_horizontal_segment[side]}")
            print(f" Bounds of the graph: min x: {min([data['pos'][0] for _, data in latest_graph.nodes(data=True)])} max x: {max([data['pos'][0] for _, data in latest_graph.nodes(data=True)])}")
            print(f"\n{side} Intersect: {intersect}\n")

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

    def _get_horizontal_segment_ped(self, horizontal_nodes_top, horizontal_nodes_bottom, latest_graph, validation=True):
        """
        Get the entire horizontal pedestrian segment of the corridor.
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
                middle_y = (bottom_pos[1] + top_pos[1]) / 2 #TODO: Change to interpolation method
                middle_pos = (middle_x, middle_y)

                # sanitize the id 
                cid = cid.replace(":", "") # Do not use a space
                middle_node_id = f"{cid}_mid"
                # Add the new middle node to the networkx graph
                self.iterative_networkx_graph.add_node(middle_node_id, pos=middle_pos, type='middle', width=3.0) # At reset, the default width of 3.0 is used

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
    
    def _get_original_veh_edge_config(self):
        """
        Get the original vehicle edge config from the original XML component files.
        """

        horizontal_edges_veh= {
        'top': ['-16666012#2', '-16666012#3', '-16666012#4', '-16666012#5', 
                                '-16666012#6', '-16666012#7', '-16666012#9', '-16666012#11', 
                                '-16666012#12', '-16666012#13', '-16666012#14', '-16666012#15', 
                                '-16666012#16', '-16666012#17'],
        'bottom': ['16666012#2', '16666012#3', '16666012#4', '16666012#5',
                                    '16666012#6', '16666012#7', '16666012#9', '16666012#11',
                                    '16666012#12', '16666012#13', '16666012#14', '16666012#15',
                                    '16666012#16', '16666012#17']
                                }
        
        node_file = f'{self.component_dir}/original.nod.xml'
        node_tree = ET.parse(node_file)
        node_root = node_tree.getroot()

        edge_file = f'{self.component_dir}/original.edg.xml'
        edge_tree = ET.parse(edge_file)
        edge_root = edge_tree.getroot()

        horizontal_edges_veh_original_data = {
            'top': {},
            'bottom': {}
        }

        for direction in ['top', 'bottom']:
            for edge in edge_root.findall('edge'):
                id = edge.get('id')
                if id in horizontal_edges_veh[direction]:
                    from_node = edge.get('from')
                    from_node_data = node_root.find(f'node[@id="{from_node}"]')
                    # Convert coordinates to float
                    from_x = float(from_node_data.get('x'))
                    from_y = float(from_node_data.get('y'))

                    to_node = edge.get('to')
                    to_node_data = node_root.find(f'node[@id="{to_node}"]')
                    # Convert coordinates to float
                    to_x = float(to_node_data.get('x'))
                    to_y = float(to_node_data.get('y'))

                    horizontal_edges_veh_original_data[direction][id] = {
                        'from_x': from_x,
                        'from_y': from_y, 
                        'to_x': to_x,
                        'to_y': to_y
                    }

        return horizontal_edges_veh_original_data
    
    def _update_xml_files(self, networkx_graph, iteration):
        """
        Update the XML component files to reflect the current state of the networkx graph.
        For base, use the "original" XML component files. For other iterations, use the "base" XML component files as a foundation and add/ remove elements.
        Iterative component files are saved in components directory.
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

        # Parse the XML files
        prefix = "original" if iteration == 'base' else "iteration_base" # Every iteration will have the same base XML files.

        node_file = f'{self.component_dir}/{prefix}.nod.xml'
        node_tree = ET.parse(node_file)
        node_root = node_tree.getroot()

        edge_file = f'{self.component_dir}/{prefix}.edg.xml'
        edge_tree = ET.parse(edge_file)
        edge_root = edge_tree.getroot()

        connection_file = f'{self.component_dir}/{prefix}.con.xml'
        connection_tree = ET.parse(connection_file)
        connection_root = connection_tree.getroot() # This will be modified by get_new_veh_edges_connections

        traffic_light_file = f'{self.component_dir}/{prefix}.tll.xml'
        traffic_light_tree = ET.parse(traffic_light_file)
        traffic_light_root = traffic_light_tree.getroot()

        type_file = f'{self.component_dir}/{prefix}.typ.xml'
        type_tree = ET.parse(type_file)

        # Find ALL the nodes and edges in the XML component files (nod.xml and edg.xml)
        nodes_in_xml = { n.get('id'): n for n in node_root.findall('node') } # save the node element itself.
        edges_in_xml = { (e.get('from'), e.get('to')): e for e in edge_root.findall('edge') } # save the from, to nodes and edge element.

        # Find PEDESTRIAN nodes and edges in the XML component .edg file.
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
        # print(f"Pedestrian nodes in XML: {pedestrian_nodes_in_xml}\n")
        # print(f"Pedestrian nodes in graph: {pedestrian_nodes_in_graph}\n")
        # print(f"Pedestrian edges in XML: {list(pedestrian_edges_in_xml.keys())}\n")
        # print(f"Pedestrian edges in graph: {set(networkx_graph.edges())}\n")

        # Remove PEDESTRIAN nodes that are in XML component file but not in networkx graph.
        potential_nodes_to_remove = pedestrian_nodes_in_xml - pedestrian_nodes_in_graph
        # print(f"Potential Nodes to remove: Total: {len(potential_nodes_to_remove)},\n {potential_nodes_to_remove}\n")

        # Some edges may still access the nodes that are in potential_nodes_to_remove.
        # Find the edges that still access the nodes that are in potential_nodes_to_remove.
        edges_in_xml_that_access_removal_nodes = {}
        for (f, t) in edges_in_xml:
            if f in potential_nodes_to_remove or t in potential_nodes_to_remove:
                edges_in_xml_that_access_removal_nodes[(f, t)] = edges_in_xml[(f, t)] # These can be vehicle edges as well.

        # print(f"Edges in XML that still access the potential removal nodes: Total: {len(edges_in_xml_that_access_removal_nodes)}")
        # for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
        #     print(f"Edge: {f} -> {t}")
        #     print(f"Edge attributes: {e.attrib}\n")

        # In the edges that access nodes in potential_nodes_to_remove, some of the edges are vehicle edges (For e.g., when the old TL was removed).
        vehicle_edges_that_access_removal_nodes = {}
        for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
            e_type = e.get('type')
            disallow = e.get('disallow', '')
            if e_type == 'highway.tertiary' and 'pedestrian' in disallow: # vehicle edge attributes: highway.tertiary and disallowed pedestrian
                vehicle_edges_that_access_removal_nodes[(f, t)] = e
        # print(f"Vehicle edges that access removal nodes: Total: {len(vehicle_edges_that_access_removal_nodes)},\n {vehicle_edges_that_access_removal_nodes}\n")

        # Get all nodes that appear in vehicle edges
        nodes_in_vehicle_edges = set()
        for f, t in vehicle_edges_that_access_removal_nodes.keys():
            nodes_in_vehicle_edges.add(f)
            nodes_in_vehicle_edges.add(t)
        # print(f"Potential nodes to be removed: {potential_nodes_to_remove}\n Nodes in vehicle edges: {nodes_in_vehicle_edges}\n")

        # The nodes that appear in vehicle edges can be removed because they are not needed for the pedestrian network. Remove them
        pedestrian_nodes_to_remove = potential_nodes_to_remove - nodes_in_vehicle_edges
        # print(f"Actual pedestrian nodes to remove: Total: {len(pedestrian_nodes_to_remove)},\n {pedestrian_nodes_to_remove}\n")

        # Some pedestrian edges (at crossings) link to vehicle edges. Remove the pedestrian edges that are not linked to the vehicle edges.
        pedestrian_edges_to_remove = {}
        for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
            if (f, t) not in vehicle_edges_that_access_removal_nodes:
                pedestrian_edges_to_remove[(f, t)] = e
        # print(f"Actual pedestrian edges to remove: Total: {len(pedestrian_edges_to_remove)}, \n {pedestrian_edges_to_remove}\n")

        # Removing selected nodes and edges
        for node_id in pedestrian_nodes_to_remove:
            if node_id in nodes_in_xml:
                node_root.remove(nodes_in_xml[node_id]) # remove from nod component file
                del nodes_in_xml[node_id] # remove from dictionary

        for (f, t) in pedestrian_edges_to_remove:
            if (f, t) in edges_in_xml:
                edge_root.remove(edges_in_xml[(f, t)]) # remove from edg component file
                del edges_in_xml[(f, t)] # remove from dictionary

        # Before new nodes are added.
        # All the nodes with tl other than default tl need to have type="dead_end" and tl attribute removed.
        default_tl = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # By default in base, there will the one TL at the left intersection. present.
        for node in node_root.findall('node'):
            tl_name = node.get('tl')
            if tl_name:
                if tl_name not in default_tl:
                    node.set('type', 'dead_end')
                    del node.attrib['tl']

        # Find the pedestrian nodes to add (present in networkx graph but not in XML component file) i.e., end nodes and middle nodes
        # In iterations other than base i.e., in iteration base, there will be no new nodes to add.
        # For regular nodes: <node id=" " x=" " y=" " />
        # For the nodes with type "middle": also add attributes: type = "traffic_light" and tl = "node_id"
        node_ids_to_add = pedestrian_nodes_in_graph - set(nodes_in_xml.keys())
        middle_nodes_to_add = []
        print(f"\nDEBUG_XML: Nodes to add: {node_ids_to_add}")

        for nid in node_ids_to_add:
            node_data = networkx_graph.nodes[nid]
            x, y = node_data['pos']
            n_type = node_data.get('type', 'regular')
            attribs = {'id': nid, 'x': str(round(x, 2)), 'y': str(round(y, 2))}

            # --- DEBUG PRINT ---
            print(f"DEBUG_XML_NODE_ADD: Processing node ID: {nid}")
            print(f"  -> Node data: {node_data}")
            print(f"  -> x={x}, y={y}, type={n_type}")
            # --- END DEBUG ---

            if n_type == 'regular':
                attribs['type'] = 'dead_end'
            elif n_type == 'middle':
                middle_nodes_to_add.append(nid)
                attribs['type'] = 'traffic_light'
                attribs['tl'] = nid # Make sure nid is a string here (it should be from graph)

            # --- DEBUG PRINT ---
            print(f"  -> Final node attributes for ET.Element: {attribs}")
            # Check types before creating element
            for key, value in attribs.items():
                if not isinstance(value, str):
                    print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
            # --- END DEBUG ---

            new_node = ET.Element('node', attribs)
            new_node.tail = "\n\t"
            node_root.append(new_node)
            nodes_in_xml[nid] = new_node

        # Find the edges to add (present in networkx graph but not in XML component file).
        ped_edges_to_add = set(networkx_graph.edges()) - set(edges_in_xml.keys()) # These are all pedestrian edges.
        ped_edges_to_add = list(ped_edges_to_add)
        # print(f"\nPedestrian edges to add: Total: {len(ped_edges_to_add)},\n {ped_edges_to_add}\n")

        # The edge could be from a type = "regular" node to a type = "regular" node or from a type = "regular" node to a type = "middle" node (crossing).
        for (f, t) in ped_edges_to_add:
            # Do Regular to Regular and Regular to Middle need some different treatment?
            edge_data = networkx_graph.get_edge_data(f, t)
            edge_id = edge_data.get('id', f'edge_{f}_{t}') # Get it from the networkx graph.
            width = edge_data.get('width', None) # There should be a width for all edges.

            # --- DEBUG PRINT ---
            print(f"DEBUG_XML_PED_EDGE_ADD: Processing edge: ({f}, {t})")
            print(f"  -> Edge data from graph: {edge_data}")
            print(f"  -> Extracted edge_id={edge_id}, width={width} (Type: {type(width)})")
            if width is None:
                print("  -> WARNING: width is None, using default string '2.0' for XML.")
                str_width = '2.0' # Provide a default if None
            else:
                str_width = str(width)
            # --- END DEBUG ---

            edge_attribs = {
                'id': edge_id, # Should be string
                'from': f,     # Should be string (node ID)
                'to': t,       # Should be string (node ID)
                'name': 'Iterative addition',
                'priority': '1',
                'type': 'highway.footway',
                'numLanes': '1',
                'speed': '2.78', # default
                'spreadType': 'center',
                'width': str_width, # Use the safe string width
                'allow': 'pedestrian'
            }

            # --- DEBUG PRINT ---
            print(f"  -> Final edge attributes for ET.Element: {edge_attribs}")
            # Check types before creating element
            for key, value in edge_attribs.items():
                if not isinstance(value, str):
                    print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
            # --- END DEBUG ---

            # positions of f and t nodes
            f_data = networkx_graph.nodes[f]
            t_data = networkx_graph.nodes[t]
            f_x, f_y = round(f_data['pos'][0], 2), round(f_data['pos'][1], 2)
            t_x, t_y = round(t_data['pos'][0], 2), round(t_data['pos'][1], 2)
            shape = f'{f_x},{f_y} {t_x},{t_y}' # shape is inherently a string

            # --- DEBUG PRINT ---
            print(f"  -> Lane shape: {shape}, Lane width: {str_width}")
            # --- END DEBUG ---

            edge_element = ET.Element('edge', edge_attribs)
            edge_element.text = "\n\t\t"

            lane_attribs = { # Attributes for the lane element
                'index':'0',
                'allow':'pedestrian',
                'width': str_width, # Use safe string width again
                'speed':'2.78',
                'shape':shape
            }

            # --- DEBUG PRINT ---
            print(f"  -> Lane attributes for ET.SubElement: {lane_attribs}")
            # Check types before creating element
            for key, value in lane_attribs.items():
                if not isinstance(value, str):
                    print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
            # --- END DEBUG ---

            lane_element = ET.SubElement(edge_element, 'lane', **lane_attribs) # Use dictionary unpacking
            lane_element.text = "\n\t\t\t"

            param_element = ET.SubElement(lane_element, 'param', key='origId', value=edge_id) # edge_id should be string
            param_element.tail = "\n\t\t"
            lane_element.tail = "\n\t"
            edge_element.tail = "\n\t"
            edge_root.append(edge_element)

        # Every middle node (present in middle_nodes_to_add) falls on a certain vehicle edge. Split the vehicle edges into two new edges.
        # The new edge names have left and right attached to the old names (the new edges inherit respective portions of the edge shape and lane shape property of the old edge)
        # This happens iteratively (because multiple middle nodes may fall on the same vehicle edge) and is a bit complex.
        # NOTE: We pass connection_root here, which might be modified inside get_new_veh_edges_connections
        print(f"DEBUG_XML: Calling get_new_veh_edges_connections with middle_nodes_to_add: {middle_nodes_to_add}")
        old_veh_edges_to_remove, new_veh_edges_to_add, updated_conn_root, m_node_mapping = get_new_veh_edges_connections(middle_nodes_to_add,
                                                                                                            networkx_graph,
                                                                                                            f'{self.component_dir}/original.edg.xml',
                                                                                                            f'{self.component_dir}/original.nod.xml',
                                                                                                            connection_root) # Pass the original root
        # --- DEBUG PRINT ---
        print(f"DEBUG_XML_VEH_SPLIT: Results from get_new_veh_edges_connections:")
        print(f"  -> old_veh_edges_to_remove: {old_veh_edges_to_remove}")
        # Printing the full new_veh_edges_to_add and m_node_mapping might be verbose, print summary or specific problematic parts if needed
        print(f"  -> new_veh_edges_to_add keys: {new_veh_edges_to_add.keys() if new_veh_edges_to_add else 'None'}")
        print(f"  -> m_node_mapping keys: {m_node_mapping.keys() if m_node_mapping else 'None'}")
        print(f"  -> connection_root was potentially updated in place. Current root tag: {updated_conn_root.tag if updated_conn_root is not None else 'None'}")
        # --- END DEBUG ---

        # It seems 'updated_conn_root' might be the new root object for the connection tree
        # Let's assume connection_tree's root should now be updated if the function returned a new one
        if updated_conn_root is not None and updated_conn_root is not connection_root:
            print("DEBUG_XML: Updating connection_tree root with the one returned from get_new_veh_edges_connections.")
            connection_tree._setroot(updated_conn_root)
            connection_root = updated_conn_root # Update local reference too
        elif updated_conn_root is None:
            print("WARNING: get_new_veh_edges_connections returned None for updated_conn_root")
            # Handle this case - maybe raise error or use original connection_root cautiously

        # print(f"old_veh_edges_to_remove: {old_veh_edges_to_remove}\n")
        # print(f"new_veh_edges_to_add: {new_veh_edges_to_add}\n")

        # Add the new edges (each edge has a single nested lane) to the edge file. The width is the default road width.
        for direction in ['top', 'bottom']:
            if direction in new_veh_edges_to_add: # Check if direction exists
                for edge_id, edge_data in new_veh_edges_to_add[direction].items():
                    # --- DEBUG PRINT ---
                    print(f"DEBUG_XML_VEH_EDGE_ADD: Processing new vehicle edge: {edge_id} (Direction: {direction})")
                    print(f"  -> Edge data: {edge_data}")
                    if edge_data is None:
                        print("   -> ERROR: edge_data is None. Skipping.")
                        continue
                    # --- END DEBUG ---

                    edge_attribs = {
                        'id': edge_id,
                        'from': edge_data.get('from'), # Use .get for safety
                        'to': edge_data.get('to'),     # Use .get for safety
                        'name': "Craver Road Iterative Addition",
                        'priority': "10",
                        'type': "highway.tertiary",
                        'numLanes': "1",
                        'speed': "8.94",
                        'disallow': "pedestrian tram rail_urban rail rail_electric rail_fast ship cable_car subway"
                    }

                    # --- DEBUG PRINT ---
                    print(f"  -> Final edge attributes for ET.Element: {edge_attribs}")
                    # Check types before creating element
                    for key, value in edge_attribs.items():
                        if value is None:
                            print(f"    -> WARNING: Attribute '{key}' has None value!")
                        elif not isinstance(value, str):
                            print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                    # --- END DEBUG ---

                    edge_element = ET.Element('edge', edge_attribs)
                    edge_element.text = "\n\t\t"

                    lane_attribs = { # Attributes for the lane element
                        'index':'0',
                        'disallow':"pedestrian tram rail_urban rail rail_electric rail_fast ship cable_car subway",
                        'speed':"8.94",
                    }

                    # --- DEBUG PRINT ---
                    print(f"  -> Lane attributes for ET.SubElement: {lane_attribs}")
                    # Check types before creating element
                    for key, value in lane_attribs.items():
                        if not isinstance(value, str):
                            print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                    # --- END DEBUG ---

                    lane_element = ET.SubElement(edge_element, 'lane', **lane_attribs)

                    lane_element.text = "\n\t\t\t"
                    param_val = edge_id.split('#')[0].replace('-', '') # Should be string

                    # --- DEBUG PRINT ---
                    print(f"  -> Param element value: {param_val}")
                    # --- END DEBUG ---

                    param_element = ET.SubElement(lane_element, 'param', key='origId', value=param_val)
                    param_element.tail = "\n\t\t"
                    lane_element.tail = "\n\t"
                    edge_element.tail = "\n\t"

                    edge_root.append(edge_element)
            else:
                print(f"DEBUG_XML_VEH_EDGE_ADD: No new vehicle edges to add for direction '{direction}'.")


        # For TL logics,
        # TL logics should come before the connections. (https://github.com/eclipse-sumo/sumo/issues/6160)
        # In order to do this, we first remove all existing TL logics except the default one.
        # We collect the connections associated with default TL and remove all connections.
        # TL 1. Remove all TLs and except the default one.
        tls_to_remove = []
        for tl in traffic_light_root.findall('tlLogic'):
            if tl.get('id') not in default_tl:
                tls_to_remove.append(tl)
        for tl in tls_to_remove:
            traffic_light_root.remove(tl)

        # TL 2. Remove all connections and store the default ones.
        tl_connections_to_add = [] # collect the connection elements.
        connections_to_remove_from_con_file = [] # Connections except the default TL should be removed from the connections file as well.
        print(f"DEBUG_XML_TL: Processing connections within TLL file...")
        for conn in traffic_light_root.findall('connection'):
            tl_id_for_conn = conn.get('tl')
            print(f"  -> Found TLL connection from={conn.get('from')} to={conn.get('to')} for tl={tl_id_for_conn}")
            traffic_light_root.remove(conn) # remove from the TLL file whether its default or not. We will add it back later.
            if tl_id_for_conn in default_tl:
                print(f"    -> Keeping (to re-add later) as it belongs to default TL.")
                tl_connections_to_add.append(conn)
            else:
                print(f"    -> Marking for removal from .con file.")
                # Store the element itself or its identifying attributes
                connections_to_remove_from_con_file.append({'from': conn.get('from'), 'to': conn.get('to'), 'fromLane': conn.get('fromLane'), 'toLane': conn.get('toLane')})


        # The TLL file connections contains connections between edges that are left and right of every midde node.
        # Due to split of split, the names of these edges may not be symmetrical (i.e., just replace left with right and vice versa wont work).
        # Use linkIndex 0 for connecting -ve direction and linkIndex 1 for connecting +ve direction.
        print(f"DEBUG_XML_TL: Creating new TL and regular connections based on m_node_mapping...")
        for direction in ['top', 'bottom']:
            if m_node_mapping: # Check if mapping exists
                for tl_id, mapping_data in m_node_mapping.items(): # m_node is the tl_id
                    linkindex = 0 if direction == 'top' else 1 # Top is -ve direction and bottom is +ve direction.
                    print(f"DEBUG_XML_TL_CONN_ADD: Processing TL connection for tl_id={tl_id}, direction={direction}")
                    print(f"  -> Mapping data for direction: {mapping_data.get(direction, 'MISSING')}") # Check if key exists

                    if mapping_data and direction in mapping_data: # Check before accessing
                        map_dir_data = mapping_data[direction]
                        from_edge = map_dir_data.get('from')
                        to_edge = map_dir_data.get('to')

                        # --- DEBUG PRINT ---
                        print(f"  -> Extracted from_edge='{from_edge}', to_edge='{to_edge}' for tl_id='{tl_id}'")
                        if from_edge is None or to_edge is None:
                            print("    -> ERROR: from_edge or to_edge is None in mapping_data. Skipping connection creation.")
                            continue
                        # --- END DEBUG ---

                        # These connections should be present in both the TLL and connections files (using left as from and right as to).
                        # TL 3. Add the new connections.
                        tl_conn_attribs = {'from': from_edge, 'to': to_edge, 'fromLane': "0", 'toLane': "0", 'tl': tl_id, 'linkIndex': str(linkindex)} # Since inside the corridor, there is only one lane.

                        # --- DEBUG PRINT ---
                        print(f"  -> TL Connection attributes for TLL file: {tl_conn_attribs}")
                        # Check types before creating element
                        for key, value in tl_conn_attribs.items():
                            if not isinstance(value, str):
                                print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                        # --- END DEBUG ---

                        tl_conn_element = ET.Element('connection', tl_conn_attribs)
                        tl_connections_to_add.append(tl_conn_element) # Add to list for later insertion into TLL

                        conn_attribs = {'from': from_edge, 'to': to_edge, 'fromLane': "0", 'toLane': "0"} # Since inside the corridor, there is only one lane.

                        # --- DEBUG PRINT ---
                        print(f"  -> Regular Connection attributes for CON file: {conn_attribs}")
                        # Check types before creating element
                        for key, value in conn_attribs.items():
                            if not isinstance(value, str):
                                print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                        # --- END DEBUG ---

                        conn_element = ET.Element('connection', conn_attribs)
                        conn_element.text = None  # Ensure there's no text content
                        conn_element.tail = "\n\t" # Adjusted tail for .con file format
                        if connection_root is not None:
                            connection_root.append(conn_element) # Append directly to connection root
                        else:
                            print("   -> ERROR: connection_root is None, cannot append connection element!")

                    else:
                        print(f"  -> WARNING: Missing mapping data or direction '{direction}' in tl_id '{tl_id}'")
            else:
                print(f"DEBUG_XML_TL_CONN_ADD: m_node_mapping is empty or None. Skipping connection creation.")


        # For the crossing tags in the Conn file ( which also dont need to be changed iteratively). # The width here needs to come from the model.
        # They are already updated while obtaining the new edges. Nothing to do here? Check get_new_veh_edges_connections.
        # Assumes get_new_veh_edges_connections potentially modified connection_root in place or returned the updated one.

        # Whereas for the crossing tags,
        # First remove all except the default ones. Then add the new ones here by making use of new_veh_edges_to_add.
        default_crossings = default_tl + ['cluster_172228408_9739966907_9739966910', '9687187500', '9687187501'] # associated with ids 0 and 10.
        print(f"DEBUG_XML_CROSSING: Removing non-default crossings from connection root...")
        if connection_root is not None:
            crossings_to_remove = []
            for crossing in connection_root.findall('crossing'):
                node_id = crossing.get('node')
                print(f"  -> Found crossing for node: {node_id}")
                if node_id not in default_crossings:
                    print(f"    -> Marking for removal (not in default list: {default_crossings})")
                    crossings_to_remove.append(crossing)
            for crossing in crossings_to_remove:
                connection_root.remove(crossing)
        else:
            print("   -> ERROR: connection_root is None, cannot remove crossings!")


        # Then deal with the existing old crossings that refer to the old edges which have been split.
        # Can be done manually.. as in -> if the leftmost edge has been split then the intersection should now refer to the new edge.
        min_x, max_x = float('inf'), float('-inf')
        leftmost_new, rightmost_new = '', ''
        if 'top' in new_veh_edges_to_add: # Check if 'top' exists
            for edge_id, edge_data in new_veh_edges_to_add['top'].items(): # One of the counterparts (among -ve, +ve) is enough.
                if edge_data: # Check if edge_data is not None
                    # Also bottom has reverse direction so top is enough.
                    from_x = edge_data.get('from_x', float('inf')) # Default to prevent error if missing
                    to_x = edge_data.get('to_x', float('inf'))     # Default to prevent error if missing
                    min_x_among_nodes = min(from_x, to_x)

                    if min_x_among_nodes < min_x:
                        min_x = min_x_among_nodes
                        # Ensure edge_id format allows splitting and forming the positive counterpart
                        if '#' in edge_id:
                            leftmost_new = f'16666012#{edge_id.split("#")[1]}'
                        else:
                            print(f"WARNING: Edge ID format unexpected for leftmost edge update: {edge_id}")


                    if min_x_among_nodes > max_x and min_x_among_nodes != float('inf'): # Check against default
                        max_x = min_x_among_nodes
                        # Ensure edge_id format allows splitting and forming the positive counterpart
                        if '#' in edge_id:
                            rightmost_new = f'16666012#{edge_id.split("#")[1]}'
                        else:
                            print(f"WARNING: Edge ID format unexpected for rightmost edge update: {edge_id}")
                else:
                    print(f"WARNING: edge_data is None for edge_id {edge_id} in new_veh_edges_to_add['top']")

        # One of the counterparts (among -ve, +ve) is enough.
        self.extreme_edge_dict['leftmost']['new'] = leftmost_new
        self.extreme_edge_dict['rightmost']['new'] = rightmost_new
        print(f"DEBUG_XML_EXTREME: Updated extreme_edge_dict: {self.extreme_edge_dict}")


        # Updates to connections and crossings in connections file.
        print(f"DEBUG_XML_EXTREME: Updating connections/crossings for extreme edges...")
        if connection_root is not None:
            for direction, direction_data in self.extreme_edge_dict.items():
                old_edge = direction_data['old']
                if old_edge in old_veh_edges_to_remove:
                    new_edge = direction_data['new']
                    print(f"DEBUG_XML_EXTREME_UPDATE: Updating for direction '{direction}'. Old: {old_edge}, New: {new_edge}")
                    if not new_edge: # Check if new_edge is empty or None
                        print(f"  -> WARNING: new_edge for {direction} is empty. Skipping updates for this direction.")
                        continue

                    crossings_to_update = connection_root.findall('crossing')
                    connections_to_add_for_crossing = []

                    for crossing in crossings_to_update:
                        current_edges = crossing.get('edges')
                        print(f"  -> Checking crossing with edges: '{current_edges}'")
                        updated = False
                        new_crossing_edges = None

                        if current_edges == f'{old_edge} -{old_edge}':
                            # First, prepare a connection between the two new edges to be added.
                            conn_attrs = {'from': new_edge, 'to': f'-{new_edge}', 'fromLane': '0', 'toLane': '0'}
                            print(f"    -> Matched {old_edge} -{old_edge}. Preparing connection with attrs: {conn_attrs}")
                            connections_to_add_for_crossing.append(conn_attrs)
                            # Then, it can be updated in crossing.
                            new_crossing_edges = f'{new_edge} -{new_edge}'
                            updated = True

                        elif current_edges == f'-{old_edge} {old_edge}':
                            # First, prepare a connection between the two new edges to be added.
                            conn_attrs = {'from': f'-{new_edge}', 'to': new_edge, 'fromLane': '0', 'toLane': '0'}
                            print(f"    -> Matched -{old_edge} {old_edge}. Preparing connection with attrs: {conn_attrs}")
                            connections_to_add_for_crossing.append(conn_attrs)
                            # Then, it can be updated in crossing.
                            new_crossing_edges = f'-{new_edge} {new_edge}'
                            updated = True

                        if updated:
                            print(f"    -> Setting crossing edges to: {new_crossing_edges}")
                            # --- DEBUG PRINT ---
                            # Check types before setting attribute
                            if not isinstance(new_crossing_edges, str):
                                print(f"      -> WARNING: new_crossing_edges is not a string: {new_crossing_edges} (Type: {type(new_crossing_edges)})")
                            # --- END DEBUG ---
                            crossing.set('edges', new_crossing_edges)

                    # Add the prepared connections after iterating through crossings
                    for conn_attrs in connections_to_add_for_crossing:
                        print(f"  -> Adding connection element to connection_root with attrs: {conn_attrs}")
                        # --- DEBUG PRINT ---
                        # Check types before creating element
                        for key, value in conn_attrs.items():
                            if not isinstance(value, str):
                                print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                        # --- END DEBUG ---
                        connection_element = ET.Element('connection', conn_attrs)
                        connection_element.text = None
                        connection_element.tail = "\n\t" # Check if tail is correct for .con file
                        connection_root.append(connection_element)

        else:
            print("   -> ERROR: connection_root is None, cannot update extreme edge connections/crossings!")


        # Add new connections (between top and bottom edges) and crossings (making use of new_veh_edges_to_add).
        # All tags that refer to the old edges should now refer to the new edges (if the refering edges fall to the left, they will refer to the new left edge and vice versa)
        # They have the edges attribute (which are edges to the right) and outlineShape attribute (the shape of the crossing):

        # outlineShape seems hard to specify, lets not specify and see what it does. They mention it as optional here: https://github.com/eclipse-sumo/sumo/issues/11668
        # TODO: same node contains right and left components which creates two crossings instead of one. Find a way to avoid this (Only add the right part of the crossing).
        print(f"DEBUG_XML_CROSSING: Adding new crossings based on new vehicle edges...")
        if connection_root is not None and 'top' in new_veh_edges_to_add: # Check root and key exist
            for e1, e1_data in new_veh_edges_to_add['top'].items(): # Just looking at one direction (top) is enough.
                if e1_data and 'right' in e1.split('_')[-1]: # Add only the right part & check e1_data
                    e2 = e1.replace('-', '') # To get the bottom edge id.
                    print(f"DEBUG_XML_CROSSING_ADD: Processing new crossing based on right-side edge e1={e1}, corresponding e2={e2}")

                    middle_node = e1_data.get('new_node')
                    print(f"  -> Corresponding middle_node: {middle_node} (Type: {type(middle_node)})")

                    # IMPORTANT: Check if middle_node itself is None before using it as key
                    if middle_node is None:
                        print("  -> ERROR: middle_node is None, cannot create crossing!")
                        continue # Skip this crossing

                    # Safely get width from graph
                    width = None
                    if middle_node in networkx_graph.nodes:
                        width = networkx_graph.nodes[middle_node].get('width')
                    else:
                        print(f"  -> WARNING: middle_node '{middle_node}' not found in networkx_graph. Cannot get width.")

                    print(f"  -> Corresponding width: {width} (Type: {type(width)})")

                    # Make width a string, handle None or missing node case explicitly
                    str_width = "2.0" # Default width if missing or None
                    if width is not None:
                        str_width = str(width)
                    else:
                        print(f"  -> Using default width '{str_width}' for crossing.")

                    crossing_attribs = {
                        'node': middle_node, # middle_node is checked for None above
                        'edges': e1 + ' ' + e2, # e1, e2 should be strings
                        'priority': '1',
                        'width': str_width, # Use safe string width
                        'linkIndex': '2'
                    }

                    # --- DEBUG PRINT ---
                    print(f"  -> Final crossing attributes for ET.Element: {crossing_attribs}")
                    # Check types before creating element
                    for key, value in crossing_attribs.items():
                        if not isinstance(value, str):
                            print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                    # --- END DEBUG ---

                    crossing_element = ET.Element('crossing', crossing_attribs)
                    crossing_element.text = None  # Ensure there's no text content
                    crossing_element.tail = "\n\t" # Adjusted tail for .con file
                    connection_root.append(crossing_element)
                elif not e1_data:
                    print(f"DEBUG_XML_CROSSING_ADD: Skipping edge e1={e1} because its data is None.")

        elif connection_root is None:
            print("   -> ERROR: connection_root is None, cannot add new crossings!")
        elif 'top' not in new_veh_edges_to_add:
            print("   -> INFO: 'top' key not found in new_veh_edges_to_add, cannot add new crossings.")


        # Delete the old edges from the edg file i.e., just remove the tags with old edge ids.
        print(f"DEBUG_XML: Removing old vehicle edges from edge root...")
        edges_to_remove_from_root = []
        for edge in edge_root.findall('edge'):
            edge_id = edge.get('id')
            if edge_id in old_veh_edges_to_remove:
                print(f"  -> Marking edge '{edge_id}' for removal.")
                edges_to_remove_from_root.append(edge)
        for edge in edges_to_remove_from_root:
            edge_root.remove(edge)

        # TL 4. Add the new TL logics.
        print(f"DEBUG_XML_TL: Adding new TL Logic elements...")
        for nid in middle_nodes_to_add:
            # --- DEBUG PRINT ---
            print(f"DEBUG_XML_TL_LOGIC_ADD: Adding TL Logic for node ID: {nid}")
            if not isinstance(nid, str):
                print(f"  -> WARNING: Node ID (TL ID) is not a string: {nid} (Type: {type(nid)})")
            # --- END DEBUG ---
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
            phase4.tail = "\n\t" # Tail for the last phase before closing tag

            tlLogic_element.tail = "\n\t" # Tail for the tlLogic element itself
            traffic_light_root.append(tlLogic_element)

        # TL 5. Add all the new and default connections back to the TLL file.
        print(f"DEBUG_XML_TL: Re-adding collected TL connection elements to TLL root...")
        for conn in tl_connections_to_add:
            # --- DEBUG PRINT ---
            print(f"DEBUG_XML_TL_CONN_RE_ADD: Re-adding TL connection element with attributes: {conn.attrib}")
            # Check types before adding element back (should be okay if checked during creation)
            for key, value in conn.attrib.items():
                if not isinstance(value, str):
                    print(f"  -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
            # --- END DEBUG ---
            conn.text = None
            conn.tail = "\n\t"
            traffic_light_root.append(conn)

        # TL 6. The default crossings in TL (that were kept above) may still refer to the old edges.
        # In addition, there may also be a connection of the -ve and +ve sides of the old edges.
        print(f"DEBUG_XML_TL: Updating default TL connections if their edges were split...")
        for direction, direction_data in self.extreme_edge_dict.items():
            old_edge = direction_data['old']
            if old_edge in old_veh_edges_to_remove:
                new_edge = direction_data['new']
                print(f"DEBUG_XML_DEFAULT_TL_CONN_UPDATE: Updating default TL connections for Old: {old_edge}, New: {new_edge}")
                if not new_edge:
                    print(f"  -> WARNING: new_edge for {direction} is empty. Skipping update for default TL connections.")
                    continue

                for conn in traffic_light_root.findall('connection'):
                    print(f"  -> Checking TL connection element with attributes: {conn.attrib}") # Before modification
                    modified = False
                    current_from = conn.get('from')
                    current_to = conn.get('to')

                    if current_from == old_edge: # positive
                        print(f"    -> Updating 'from' from {old_edge} to {new_edge}")
                        conn.set('from', new_edge)
                        modified = True
                    if current_from == f"-{old_edge}": # negative
                        print(f"    -> Updating 'from' from -{old_edge} to -{new_edge}")
                        conn.set('from', f"-{new_edge}")
                        modified = True
                    if current_to == old_edge: # positive
                        print(f"    -> Updating 'to' from {old_edge} to {new_edge}")
                        conn.set('to', new_edge)
                        modified = True
                    if current_to == f"-{old_edge}": # negative
                        print(f"    -> Updating 'to' from -{old_edge} to -{new_edge}")
                        conn.set('to', f"-{new_edge}")
                        modified = True

                    if modified:
                        print(f"    --> Modified TL connection attributes: {conn.attrib}") # After modification
                        # --- DEBUG PRINT ---
                        # Check types after modification
                        for key, value in conn.attrib.items():
                            if not isinstance(value, str):
                                print(f"      -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                        # --- END DEBUG ---


        # Respective changes to the connections file.
        # All the connections present in the TLL file should also be present in the connections file. But the connection file will have more of them.
        # In iteration base, there will be a bunch of connections to remove from original file (remove connections with the same from and to edges).
        # all_conn_file_connections = [(conn.get('from'), conn.get('to')) for conn in connection_root.findall('connection')]
        # print(f"connection Before removal: Total: {len(all_conn_file_connections)},\n {all_conn_file_connections}\n")

        # Look at the same from and to edges in the connections file and remove them.
        # These were connections associated with non-default TLs in the original TLL file.
        print(f"DEBUG_XML_CON: Removing connections from .con file that belonged to removed TLs...")
        if connection_root is not None:
            conns_to_remove_from_con_root = []
            for conn_element in connection_root.findall('connection'):
                f = conn_element.get('from')
                t = conn_element.get('to')
                fl = conn_element.get('fromLane')
                tlane = conn_element.get('toLane')
                # Check if this connection matches one marked for removal
                for removal_marker in connections_to_remove_from_con_file:
                    if (f == removal_marker['from'] and
                        t == removal_marker['to'] and
                        fl == removal_marker['fromLane'] and
                        tlane == removal_marker['toLane']):
                        print(f"  -> Marking connection from={f} to={t} lanes={fl}->{tlane} for removal from .con file.")
                        conns_to_remove_from_con_root.append(conn_element)
                        break # Move to next connection element once matched

            for conn_element in conns_to_remove_from_con_root:
                connection_root.remove(conn_element)
        else:
            print("   -> ERROR: connection_root is None, cannot remove connections!")

        # Additional stuff related to edge removals.
        # If the edge (pedestrian and vehicle) is removed, then the connections to and from that edge should also be removed.
        pedestrian_edges_ids_to_remove = []
        for (f,t), edge in pedestrian_edges_to_remove.items():
            edge_id = edge.get('id')
            if edge_id:
                pedestrian_edges_ids_to_remove.append(edge_id)

        print(f"DEBUG_XML_CON: Removing connections involving removed pedestrian edges: {pedestrian_edges_ids_to_remove}")

        if connection_root is not None:
            conns_involving_removed_edges = []
            for conn in connection_root.findall('connection'):
                f = conn.get('from')
                t = conn.get('to')
                if f in pedestrian_edges_ids_to_remove or t in pedestrian_edges_ids_to_remove:
                    print(f"  -> Marking connection from={f} to={t} for removal (involves removed ped edge).")
                    conns_involving_removed_edges.append(conn)
            for conn in conns_involving_removed_edges:
                connection_root.remove(conn)
        else:
            print("   -> ERROR: connection_root is None, cannot remove connections involving removed ped edges!")


        iteration_prefix = f'{self.component_dir}/iteration_{iteration}'
        print(f"\nDEBUG_XML: Writing final component XML files with prefix: {iteration_prefix}")
        node_tree.write(f'{iteration_prefix}.nod.xml', encoding='utf-8', xml_declaration=True)
        edge_tree.write(f'{iteration_prefix}.edg.xml', encoding='utf-8', xml_declaration=True)
        # Use the potentially updated connection_tree (check if its root was replaced)
        connection_tree.write(f'{iteration_prefix}.con.xml', encoding='utf-8', xml_declaration=True)
        type_tree.write(f'{iteration_prefix}.typ.xml', encoding='utf-8', xml_declaration=True)
        traffic_light_tree.write(f'{iteration_prefix}.tll.xml', encoding='utf-8', xml_declaration=True)
        print("DEBUG_XML: Component files written.")


        # Generate the final net file using netconvert
        output_file = f'{self.network_dir}/network_iteration_{iteration}.net.xml'
        netconvert_log_file = f'simulation/netconvert_log_{iteration}.txt' # Unique log per iteration
        command = (
            f"netconvert "
            f"--node-files={iteration_prefix}.nod.xml "
            f"--edge-files={iteration_prefix}.edg.xml "
            f"--connection-files={iteration_prefix}.con.xml "
            f"--type-files={iteration_prefix}.typ.xml "
            f"--tllogic-files={iteration_prefix}.tll.xml "
            f"--output-file={output_file} "
            f"--log={netconvert_log_file} "
            f"--xml-validation never " # Temporarily disable strict validation if needed for debugging netconvert errors
            f"--verbose" # Get more output from netconvert
        )
        print(f"DEBUG_XML: Running netconvert command:\n{command}")


        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                # Using PIPE for stdout/stderr allows capturing them if needed
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=30) # Added timeout
                print(f"Netconvert stdout (attempt {attempt+1}):\n{result.stdout}")
                if result.stderr:
                    # Warnings are often printed to stderr by netconvert
                    print(f"Netconvert stderr/warnings (attempt {attempt+1}):\n{result.stderr}")
                print(f"Netconvert finished successfully on attempt {attempt+1}.")
                break # Success
            except subprocess.CalledProcessError as e:
                print(f"Error running netconvert (attempt {attempt + 1}/{max_attempts}): {e}")
                print(f"Netconvert return code: {e.returncode}")
                print(f"Netconvert stdout:\n{e.stdout}")
                print(f"Netconvert stderr:\n{e.stderr}")
                attempt += 1
                if attempt == max_attempts:
                    print("Failed all attempts to run netconvert. Check logs and XML files.")
                    print(f"Check netconvert log: {netconvert_log_file}")
                    print(f"Check component files prefixed with: {iteration_prefix}")
                    # Optionally: Copy problematic files for inspection
                    # shutil.copy(f'{iteration_prefix}.nod.xml', f'{iteration_prefix}.nod.xml.failed')
                    # ... etc for other files ...
                    raise # Re-raise the exception after logging
            except subprocess.TimeoutExpired as e:
                print(f"Netconvert timed out (attempt {attempt + 1}/{max_attempts}): {e}")
                print(f"Netconvert stdout:\n{e.stdout}")
                print(f"Netconvert stderr:\n{e.stderr}")
                attempt += 1
                if attempt == max_attempts:
                    print("Failed all attempts to run netconvert due to timeout.")
                    raise

    # def _update_xml_files(self, networkx_graph, iteration):
    #     """
    #     Update the XML component files to reflect the current state of the networkx graph. 
    #     For base, use the "original" XML component files. For other iterations, use the "base" XML component files as a foundation and add/ remove elements.
    #     Iterative component files are saved in components directory.
    #     Iterative net files are saved in network_iterations directory.

    #     Networkx graph will already have:
    #       - End nodes with position values that come from the proposal.
    #       - Middle nodes to create traffic lights. Every proposal will have nodes with id _mid. 
    #             - The _mid nodes need to be connected to vehicle edges on either side as well. 
    #             - For every _mid node, a TL logic needs to be added to the traffic_light XML file.

    #     For iteration = base, the end nodes are already present in the XML component files.
    #     For other iterations, the end nodes will require reconstruction.

    #     For the nodes and edges related to pedestrian network:
    #         Remove them from the XML component files: If they don't exist in networkx graph.
    #         Add them to the XML component files: If they exist in networkx graph but don't exist in component XML.

    #     Node attributes: 
    #         - <node id=" " x=" " y=" " type=" " />
    #         - For the middle nodes, type will "traffic_light" and an attribute tl =" " with the value same as id.
    #         - For the end nodes, type will be "dead_end"
    #     Edge attributes: 
    #         - From the middle node to end nodes of type "highway.footway"
    #         - From middle node to vehicle nodes of type "highway.tertiary"
    #         - Both of these are needed because the traffic light is coordinating vehicles and pedestrians in the crosswalk.
    #         - <edge id=" " from=" " to=" " priority="1" type="highway.footway" numLanes="1" speed="2.78" shape=" " spreadType="center" width="2.0" allow="pedestrian"> 
    #         - shape seems difficult to get right.
    #         - create a nested lane element: <lane index="0" allow="pedestrian" width=" " speed=" ">
    #         - create a nested param element: <param key="origId" value=" "/>
    #         - end with </lane></edge>
    #     """

    #     # Parse the XML files
    #     prefix = "original" if iteration == 'base' else "iteration_base" # Every iteration will have the same base XML files.
        
    #     node_file = f'{self.component_dir}/{prefix}.nod.xml'
    #     node_tree = ET.parse(node_file)
    #     node_root = node_tree.getroot()

    #     edge_file = f'{self.component_dir}/{prefix}.edg.xml'
    #     edge_tree = ET.parse(edge_file)
    #     edge_root = edge_tree.getroot()

    #     connection_file = f'{self.component_dir}/{prefix}.con.xml'
    #     connection_tree = ET.parse(connection_file)
    #     connection_root = connection_tree.getroot() 

    #     traffic_light_file = f'{self.component_dir}/{prefix}.tll.xml'
    #     traffic_light_tree = ET.parse(traffic_light_file)
    #     traffic_light_root = traffic_light_tree.getroot()

    #     type_file = f'{self.component_dir}/{prefix}.typ.xml'
    #     type_tree = ET.parse(type_file)

    #     # Find ALL the nodes and edges in the XML component files (nod.xml and edg.xml)
    #     nodes_in_xml = { n.get('id'): n for n in node_root.findall('node') } # save the node element itself.
    #     edges_in_xml = { (e.get('from'), e.get('to')): e for e in edge_root.findall('edge') } # save the from, to nodes and edge element.

    #     # Find PEDESTRIAN nodes and edges in the XML component .edg file. 
    #     pedestrian_edges_in_xml = {}
    #     pedestrian_nodes_in_xml = set()
    #     for (f, t), e in edges_in_xml.items():
    #         e_type = e.get('type')
    #         allow = e.get('allow', '')
    #         if e_type in ['highway.footway', 'highway.steps'] and 'pedestrian' in allow:
    #             pedestrian_edges_in_xml[(f, t)] = e
    #             pedestrian_nodes_in_xml.update([f, t]) # From node id alone we cant differentiate between vehicle and pedestrian nodes.

    #     # Extract pedestrian nodes and edges from networkx_graph
    #     pedestrian_nodes_in_graph = set(networkx_graph.nodes())
    #     # print(f"Pedestrian nodes in XML: {pedestrian_nodes_in_xml}\n")
    #     # print(f"Pedestrian nodes in graph: {pedestrian_nodes_in_graph}\n")
    #     # print(f"Pedestrian edges in XML: {list(pedestrian_edges_in_xml.keys())}\n")
    #     # print(f"Pedestrian edges in graph: {set(networkx_graph.edges())}\n")
        
    #     # Remove PEDESTRIAN nodes that are in XML component file but not in networkx graph.
    #     potential_nodes_to_remove = pedestrian_nodes_in_xml - pedestrian_nodes_in_graph
    #     # print(f"Potential Nodes to remove: Total: {len(potential_nodes_to_remove)},\n {potential_nodes_to_remove}\n")
        
    #     # Some edges may still access the nodes that are in potential_nodes_to_remove.
    #     # Find the edges that still access the nodes that are in potential_nodes_to_remove.
    #     edges_in_xml_that_access_removal_nodes = {}
    #     for (f, t) in edges_in_xml:
    #         if f in potential_nodes_to_remove or t in potential_nodes_to_remove:
    #             edges_in_xml_that_access_removal_nodes[(f, t)] = edges_in_xml[(f, t)] # These can be vehicle edges as well.

    #     # print(f"Edges in XML that still access the potential removal nodes: Total: {len(edges_in_xml_that_access_removal_nodes)}")
    #     for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
    #         print(f"Edge: {f} -> {t}")
    #         print(f"Edge attributes: {e.attrib}\n")

    #     # In the edges that access nodes in potential_nodes_to_remove, some of the edges are vehicle edges (For e.g., when the old TL was removed).
    #     vehicle_edges_that_access_removal_nodes = {}
    #     for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
    #         e_type = e.get('type')
    #         disallow = e.get('disallow', '')
    #         if e_type == 'highway.tertiary' and 'pedestrian' in disallow: # vehicle edge attributes: highway.tertiary and disallowed pedestrian
    #             vehicle_edges_that_access_removal_nodes[(f, t)] = e
    #     # print(f"Vehicle edges that access removal nodes: Total: {len(vehicle_edges_that_access_removal_nodes)},\n {vehicle_edges_that_access_removal_nodes}\n")
        
    #     # Get all nodes that appear in vehicle edges
    #     nodes_in_vehicle_edges = set()
    #     for f, t in vehicle_edges_that_access_removal_nodes.keys():
    #         nodes_in_vehicle_edges.add(f)
    #         nodes_in_vehicle_edges.add(t)
    #     # print(f"Potential nodes to be removed: {potential_nodes_to_remove}\n Nodes in vehicle edges: {nodes_in_vehicle_edges}\n")
 
    #     # The nodes that appear in vehicle edges can be removed because they are not needed for the pedestrian network. Remove them
    #     pedestrian_nodes_to_remove = potential_nodes_to_remove - nodes_in_vehicle_edges
    #     # print(f"Actual pedestrian nodes to remove: Total: {len(pedestrian_nodes_to_remove)},\n {pedestrian_nodes_to_remove}\n")

    #     # Some pedestrian edges (at crossings) link to vehicle edges. Remove the pedestrian edges that are not linked to the vehicle edges. 
    #     pedestrian_edges_to_remove = {}
    #     for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
    #         if (f, t) not in vehicle_edges_that_access_removal_nodes:
    #             pedestrian_edges_to_remove[(f, t)] = e
    #     # print(f"Actual pedestrian edges to remove: Total: {len(pedestrian_edges_to_remove)}, \n {pedestrian_edges_to_remove}\n")

    #     # Removing selected nodes and edges
    #     for node_id in pedestrian_nodes_to_remove:
    #         if node_id in nodes_in_xml:
    #             node_root.remove(nodes_in_xml[node_id]) # remove from nod component file
    #             del nodes_in_xml[node_id] # remove from dictionary

    #     for (f, t) in pedestrian_edges_to_remove:
    #         if (f, t) in edges_in_xml:
    #             edge_root.remove(edges_in_xml[(f, t)]) # remove from edg component file
    #             del edges_in_xml[(f, t)] # remove from dictionary

    #     # Before new nodes are added.
    #     # All the nodes with tl other than default tl need to have type="dead_end" and tl attribute removed.
    #     default_tl = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # By default in base, there will the one TL at the left intersection. present.
    #     for node in node_root.findall('node'):
    #         tl_name = node.get('tl')
    #         if tl_name:
    #             if tl_name not in default_tl:
    #                 node.set('type', 'dead_end')
    #                 del node.attrib['tl']

    #     # Find the pedestrian nodes to add (present in networkx graph but not in XML component file) i.e., end nodes and middle nodes
    #     # In iterations other than base i.e., in iteration base, there will be no new nodes to add.
    #     # For regular nodes: <node id=" " x=" " y=" " />
    #     # For the nodes with type "middle": also add attributes: type = "traffic_light" and tl = "node_id" 
    #     node_ids_to_add = pedestrian_nodes_in_graph - set(nodes_in_xml.keys()) 
    #     middle_nodes_to_add = []
    #     print(f"\nNodes to add: {node_ids_to_add}")

    #     for nid in node_ids_to_add:
    #         node_data = networkx_graph.nodes[nid]
    #         x, y = node_data['pos']
    #         n_type = node_data.get('type', 'regular')
    #         attribs = {'id': nid, 'x': str(round(x, 2)), 'y': str(round(y, 2))}

    #         if n_type == 'regular':
    #             attribs['type'] = 'dead_end'
    #         elif n_type == 'middle':
    #             middle_nodes_to_add.append(nid)
    #             attribs['type'] = 'traffic_light'
    #             attribs['tl'] = nid

    #         new_node = ET.Element('node', attribs)
    #         new_node.tail = "\n\t"
    #         node_root.append(new_node)
    #         nodes_in_xml[nid] = new_node

    #     # Find the edges to add (present in networkx graph but not in XML component file).
    #     ped_edges_to_add = set(networkx_graph.edges()) - set(edges_in_xml.keys()) # These are all pedestrian edges.
    #     ped_edges_to_add = list(ped_edges_to_add)
    #     # print(f"\nPedestrian edges to add: Total: {len(ped_edges_to_add)},\n {ped_edges_to_add}\n")

    #     # The edge could be from a type = "regular" node to a type = "regular" node or from a type = "regular" node to a type = "middle" node (crossing).
    #     for (f, t) in ped_edges_to_add:
    #         # Do Regular to Regular and Regular to Middle need some different treatment?
    #         edge_data = networkx_graph.get_edge_data(f, t)
    #         edge_id = edge_data.get('id', f'edge_{f}_{t}') # Get it from the networkx graph.
    #         width = edge_data.get('width', None) # There should be a width for all edges.
    #         edge_attribs = {
    #             'id': edge_id,
    #             'from': f,
    #             'to': t,
    #             'name': 'Iterative addition',
    #             'priority': '1',
    #             'type': 'highway.footway',
    #             'numLanes': '1',
    #             'speed': '2.78', # default
    #             'spreadType': 'center',
    #             'width': str(width),
    #             'allow': 'pedestrian'
    #         }

    #         # positions of f and t nodes
    #         f_data = networkx_graph.nodes[f]
    #         t_data = networkx_graph.nodes[t]
    #         f_x, f_y = round(f_data['pos'][0], 2), round(f_data['pos'][1], 2)
    #         t_x, t_y = round(t_data['pos'][0], 2), round(t_data['pos'][1], 2)
    #         shape = f'{f_x},{f_y} {t_x},{t_y}'

    #         edge_element = ET.Element('edge', edge_attribs)
    #         edge_element.text = "\n\t\t" 

    #         lane_element = ET.SubElement(
    #             edge_element,
    #             'lane', 
    #             index='0', 
    #             allow='pedestrian',
    #             width=str(width),
    #             speed='2.78', 
    #             shape=shape)

    #         lane_element.text = "\n\t\t\t" 
    #         param_element = ET.SubElement(lane_element, 'param', key='origId', value=edge_id)
    #         param_element.tail = "\n\t\t" 
    #         lane_element.tail = "\n\t"
    #         edge_element.tail = "\n\t"
    #         edge_root.append(edge_element)

    #     # Every middle node (present in middle_nodes_to_add) falls on a certain vehicle edge. Split the vehicle edges into two new edges.
    #     # The new edge names have left and right attached to the old names (the new edges inherit respective portions of the edge shape and lane shape property of the old edge)
    #     # This happens iteratively (because multiple middle nodes may fall on the same vehicle edge) and is a bit complex.
    #     old_veh_edges_to_remove, new_veh_edges_to_add, updated_conn_root, m_node_mapping = get_new_veh_edges_connections(middle_nodes_to_add, 
    #                                                                                                      networkx_graph, 
    #                                                                                                      f'{self.component_dir}/original.edg.xml', 
    #                                                                                                      f'{self.component_dir}/original.nod.xml', 
    #                                                                                                      connection_root)
    #     # print(f"old_veh_edges_to_remove: {old_veh_edges_to_remove}\n")
    #     # print(f"new_veh_edges_to_add: {new_veh_edges_to_add}\n")

    #     # Add the new edges (each edge has a single nested lane) to the edge file. The width is the default road width.
    #     for direction in ['top', 'bottom']:
    #         for edge_id, edge_data in new_veh_edges_to_add[direction].items():
    #             edge_attribs = {
    #                 'id': edge_id,
    #                 'from': edge_data.get('from'),
    #                 'to': edge_data.get('to'),
    #                 'name': "Craver Road Iterative Addition",
    #                 'priority': "10",
    #                 'type': "highway.tertiary",
    #                 'numLanes': "1",
    #                 'speed': "8.94",
    #                 'disallow': "pedestrian tram rail_urban rail rail_electric rail_fast ship cable_car subway"
    #             }

    #             edge_element = ET.Element('edge', edge_attribs)
    #             edge_element.text = "\n\t\t"

    #             lane_element = ET.SubElement(edge_element, 
    #                                          'lane', 
    #                                          index='0', 
    #                                          disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship cable_car subway", 
    #                                          speed="8.94", 
    #                                          )
                
    #             lane_element.text = "\n\t\t\t"
    #             param_element = ET.SubElement(lane_element, 'param', key='origId', value=edge_id.split('#')[0].replace('-', '')) # remove the negative sign and #
    #             param_element.tail = "\n\t\t"
    #             lane_element.tail = "\n\t"
    #             edge_element.tail = "\n\t"

    #             edge_root.append(edge_element)
        
    #     # For TL logics,
    #     # TL logics should come before the connections. (https://github.com/eclipse-sumo/sumo/issues/6160)
    #     # In order to do this, we first remove all existing TL logics except the default one.
    #     # We collect the connections associated with default TL and remove all connections.
    #     # TL 1. Remove all TLs and except the default one.
    #     tls_to_remove = []
    #     for tl in traffic_light_root.findall('tlLogic'):
    #         if tl.get('id') not in default_tl:
    #             tls_to_remove.append(tl)
    #     for tl in tls_to_remove:
    #         traffic_light_root.remove(tl)

    #     # TL 2. Remove all connections and store the default ones.
    #     tl_connections_to_add = [] # collect the connection elements.
    #     connections_to_remove = [] # Connections except the default TL should be removed from the connections file as well.
    #     for conn in traffic_light_root.findall('connection'):
    #         traffic_light_root.remove(conn) # remove from the TLL file whether its default or not. We will add it back later.
    #         if conn.get('tl') in default_tl:
    #             tl_connections_to_add.append(conn)
    #         else:
    #             connections_to_remove.append(conn) # remove later from the connections file.

    #     # The TLL file connections contains connections between edges that are left and right of every midde node.
    #     # Due to split of split, the names of these edges may not be symmetrical (i.e., just replace left with right and vice versa wont work).
    #     # Use linkIndex 0 for connecting -ve direction and linkIndex 1 for connecting +ve direction.
    #     for direction in ['top', 'bottom']:
    #         for tl_id, mapping_data in m_node_mapping.items(): # m_node is the tl_id
    #             linkindex = 0 if direction == 'top' else 1 # Top is -ve direction and bottom is +ve direction.
                
    #             # These connections should be present in both the TLL and connections files (using left as from and right as to).
    #             # TL 3. Add the new connections.
    #             tl_conn_attribs = {'from': mapping_data[direction]['from'], 'to': mapping_data[direction]['to'], 'fromLane': "0", 'toLane': "0", 'tl': tl_id, 'linkIndex': str(linkindex)} # Since inside the corridor, there is only one lane.
    #             tl_conn_element = ET.Element('connection', tl_conn_attribs)
    #             tl_connections_to_add.append(tl_conn_element)

    #             conn_attribs = {'from': mapping_data[direction]['from'], 'to': mapping_data[direction]['to'], 'fromLane': "0", 'toLane': "0"} # Since inside the corridor, there is only one lane.
    #             conn_element = ET.Element('connection', conn_attribs)
    #             conn_element.text = None  # Ensure there's no text content
    #             conn_element.tail = "\n\t\t"
    #             updated_conn_root.append(conn_element)

    #     # For the crossing tags in the Conn file ( which also dont need to be changed iteratively). # The width here needs to come from the model. 
    #     # They are already updated while obtaining the new edges. Nothing to do here.
    #     # Whereas for the crossing tags,
    #     # First remove all except the default ones. Then add the new ones here by making use of new_veh_edges_to_add.
    #     default_crossings = default_tl + ['cluster_172228408_9739966907_9739966910', '9687187500', '9687187501'] # associated with ids 0 and 10.
    #     for crossing in updated_conn_root.findall('crossing'):
    #         if crossing.get('node') not in default_crossings:
    #             updated_conn_root.remove(crossing)
        
    #     # Then deal with the existing old crossings that refer to the old edges which have been split. 
    #     # Can be done manually.. as in -> if the leftmost edge has been split then the intersection should now refer to the new edge.
    #     min_x, max_x = float('inf'), float('-inf')
    #     leftmost_new, rightmost_new = '', ''
    #     for edge_id, edge_data in new_veh_edges_to_add['top'].items(): # One of the counterparts (among -ve, +ve) is enough.
    #         # Also bottom has reverse direction so top is enough.
    #         min_x_among_nodes = min(edge_data.get('from_x'), edge_data.get('to_x'))
    #         if min_x_among_nodes < min_x:
    #             min_x = min_x_among_nodes
    #             leftmost_new = f'16666012#{edge_id.split("#")[1]}'
    #         if min_x_among_nodes > max_x:
    #             max_x = min_x_among_nodes
    #             rightmost_new = f'16666012#{edge_id.split("#")[1]}'

    #     # One of the counterparts (among -ve, +ve) is enough.
    #     self.extreme_edge_dict['leftmost']['new'] = leftmost_new
    #     self.extreme_edge_dict['rightmost']['new'] = rightmost_new
        
    #     # Updates to connections and crossings in connections file.
    #     for direction, direction_data in self.extreme_edge_dict.items():
    #         old_edge = direction_data['old']
    #         if old_edge in old_veh_edges_to_remove:
    #             new_edge = direction_data['new']
    #             print(f"\n\nold_edge: {old_edge}, new_edge: {new_edge}\n\n")
                
    #             for crossing in updated_conn_root.findall('crossing'):
    #                 if crossing.get('edges') == f'{old_edge} -{old_edge}':
    #                     # First, a connection between the two new edges should be added.
    #                     connection_element = ET.Element('connection', {'from': new_edge, 'to': f'-{new_edge}', 'fromLane': '0', 'toLane': '0'})
    #                     connection_element.text = None  # Ensure there's no text content
    #                     connection_element.tail = "\n\t\t"
    #                     updated_conn_root.append(connection_element)
    #                     # Then, it can be updated in crossing.
    #                     crossing.set('edges', f'{new_edge} -{new_edge}')

    #                 elif crossing.get('edges') == f'-{old_edge} {old_edge}':
    #                     # First, a connection between the two new edges should be added.
    #                     connection_element = ET.Element('connection', {'from': f'-{new_edge}', 'to': new_edge, 'fromLane': '0', 'toLane': '0'})
    #                     connection_element.text = None  # Ensure there's no text content
    #                     connection_element.tail = "\n\t\t"
    #                     updated_conn_root.append(connection_element)

    #                     # Then, it can be updated in crossing.
    #                     crossing.set('edges', f'-{new_edge} {new_edge}')

        
    #     # Add new connections (between top and bottom edges) and crossings (making use of new_veh_edges_to_add).
    #     # All tags that refer to the old edges should now refer to the new edges (if the refering edges fall to the left, they will refer to the new left edge and vice versa) 
    #     # They have the edges attribute (which are edges to the right) and outlineShape attribute (the shape of the crossing): 
        
    #     # outlineShape seems hard to specify, lets not specify and see what it does. They mention it as optional here: https://github.com/eclipse-sumo/sumo/issues/11668
    #     # TODO: same node contains right and left components which creates two crossings instead of one. Find a way to avoid this (Only add the right part of the crossing).
    #     for e1, e1_data in new_veh_edges_to_add['top'].items(): # Just looking at one direction (top) is enough.
    #         if 'right' in e1.split('_')[-1]: # Add only the right part: 
    #             e2 = e1.replace('-', '') # To get the bottom edge id.
    #             print(f"e1: {e1}, e2: {e2}")

    #             # Then, a crossing element should be added with those edges.
    #             middle_node = e1_data.get('new_node')
    #             width = networkx_graph.nodes[middle_node].get('width')
    #             crossing_attribs = {'node': middle_node, 'edges': e1 + ' ' + e2, 'priority': '1', 'width': str(width), 'linkIndex': '2' } # Width/ Thickness needs to come from the model.
    #             crossing_element = ET.Element('crossing', crossing_attribs)
    #             crossing_element.text = None  # Ensure there's no text content
    #             crossing_element.tail = "\n\t\t"
    #             updated_conn_root.append(crossing_element)

    #     # Delete the old edges from the edg file i.e., just remove the tags with old edge ids.
    #     for edge in edge_root.findall('edge'):
    #         if edge.get('id') in old_veh_edges_to_remove:
    #             edge_root.remove(edge)

    #     # TL 4. Add the new TL logics.
    #     for nid in middle_nodes_to_add:
    #         tlLogic_element = ET.Element('tlLogic', id=nid, type='static', programID='0', offset='0')
    #         tlLogic_element.text = "\n\t\t" # Inside <tlLogic>: phases start at two tabs

    #         # Create phases with proper indentation
    #         phase1 = ET.SubElement(tlLogic_element, 'phase', duration='77', state='GGr')
    #         phase1.tail = "\n\t\t"
    #         phase2 = ET.SubElement(tlLogic_element, 'phase', duration='3', state='yyr') 
    #         phase2.tail = "\n\t\t"
    #         phase3 = ET.SubElement(tlLogic_element, 'phase', duration='5', state='rrG')
    #         phase3.tail = "\n\t\t"
    #         phase4 = ET.SubElement(tlLogic_element, 'phase', duration='5', state='rrr')
    #         phase4.tail = "\n\t"

    #         tlLogic_element.tail = "\n\t"
    #         traffic_light_root.append(tlLogic_element)
        
    #     # TL 5. Add all the new connections.
    #     for conn in tl_connections_to_add:
    #         conn.text = None  
    #         conn.tail = "\n\t"
    #         traffic_light_root.append(conn)

    #     # TL 6. The default crossings in TL (that were kept above) may still refer to the old edges.
    #     # In addition, there may also be a connection of the -ve and +ve sides of the old edges.
    #     for direction, direction_data in self.extreme_edge_dict.items():
    #         old_edge = direction_data['old']
    #         if old_edge in old_veh_edges_to_remove:
    #             new_edge = direction_data['new']
    #             for conn in traffic_light_root.findall('connection'):
    #                 if conn.get('from') == old_edge: # positive
    #                     conn.set('from', new_edge)
    #                 if conn.get('from') == f"-{old_edge}": # negative
    #                     conn.set('from', f"-{new_edge}") 
    #                 if conn.get('to') == old_edge: # positive
    #                     conn.set('to', new_edge)
    #                 if conn.get('to') == f"-{old_edge}": # negative
    #                     conn.set('to', f"-{new_edge}")

    #     # Respective changes to the connections file.
    #     # All the connections present in the TLL file should also be present in the connections file. But the connection file will have more of them.
    #     # In iteration base, there will be a bunch of connections to remove from original file (remove connections with the same from and to edges).
    #     # all_conn_file_connections = [(conn.get('from'), conn.get('to')) for conn in connection_root.findall('connection')]
    #     # print(f"connection Before removal: Total: {len(all_conn_file_connections)},\n {all_conn_file_connections}\n")
        
    #     # Look at the same from and to edges in the connections file and remove them.
    #     connections_to_remove_list = [(conn.get('from'), conn.get('to')) for conn in connections_to_remove]
    #     to_remove = []
    #     for conn in connection_root.findall('connection'):
    #         from_edge = conn.get('from')
    #         to_edge = conn.get('to')
    #         if (from_edge, to_edge) in connections_to_remove_list:
    #             to_remove.append(conn)
    #     for conn in to_remove:
    #         connection_root.remove(conn)

    #     # Additional stuff related to edge removals.
    #     # If the edge (pedestrian and vehicle) is removed, then the connections to and from that edge should also be removed.
    #     pedestrian_edges_to_remove_connections = []
    #     for (f,t), edge in pedestrian_edges_to_remove.items():
    #         pedestrian_edges_to_remove_connections.append(edge.get('id'))

    #     print(f"pedestrian_edges_to_remove_connections: Total: {len(pedestrian_edges_to_remove_connections)},\n {pedestrian_edges_to_remove_connections}\n")

    #     for conn in connection_root.findall('connection'):
    #         if conn.get('from') in pedestrian_edges_to_remove_connections or conn.get('to') in pedestrian_edges_to_remove_connections:
    #             connection_root.remove(conn)
        
    #     iteration_prefix = f'{self.component_dir}/iteration_{iteration}'
    #     node_tree.write(f'{iteration_prefix}.nod.xml', encoding='utf-8', xml_declaration=True)
    #     edge_tree.write(f'{iteration_prefix}.edg.xml', encoding='utf-8', xml_declaration=True)
    #     connection_tree.write(f'{iteration_prefix}.con.xml', encoding='utf-8', xml_declaration=True)
    #     type_tree.write(f'{iteration_prefix}.typ.xml', encoding='utf-8', xml_declaration=True)
    #     traffic_light_tree.write(f'{iteration_prefix}.tll.xml', encoding='utf-8', xml_declaration=True)

    #     # Generate the final net file using netconvert
    #     output_file = f'{self.network_dir}/network_iteration_{iteration}.net.xml'
    #     netconvert_log_file = f'simulation/netconvert_log.txt'
    #     command = (
    #         f"netconvert "
    #         f"--node-files={iteration_prefix}.nod.xml "
    #         f"--edge-files={iteration_prefix}.edg.xml "
    #         f"--connection-files={iteration_prefix}.con.xml "
    #         f"--type-files={iteration_prefix}.typ.xml "
    #         f"--tllogic-files={iteration_prefix}.tll.xml "
    #         f"--output-file={output_file} "
    #         f"--log={netconvert_log_file}"
    #     )


    #     max_attempts = 3
    #     attempt = 0
    #     while attempt < max_attempts:
    #         try:
    #             result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    #             if result.stderr:
    #                 print(f"Warnings/Errors from netconvert: {result.stderr}")
    #             break
    #         except subprocess.CalledProcessError as e:
    #             print(f"Error running netconvert (attempt {attempt + 1}/{max_attempts}): {e}")
    #             print("Error output:", e.stderr)
    #             attempt += 1
    #             if attempt == max_attempts:
    #                 print("Failed all attempts to run netconvert")
    #                 raise

    def _initialize_normalizers(self, pedestrian_networkx_graph):
        """
        Initialize normalizers based on the graph coordinates
        """
        # Get all horizontal segments
        horizontal_segment = self._get_horizontal_segment_ped(
            self.horizontal_nodes_top_ped, 
            self.horizontal_nodes_bottom_ped, 
            pedestrian_networkx_graph
        )
        
        offset = 2 # Add or Subtract a small buffer.

        # Find the leftmost and rightmost valid x coordinates from both top and bottom segments
        top_x_coords = list(horizontal_segment['top'].keys())
        bottom_x_coords = list(horizontal_segment['bottom'].keys())
        
        # Get the leftmost valid x (start of first segment)
        min_x = max(min(top_x_coords), min(bottom_x_coords)) + offset
        
        # Get the rightmost valid x (end of last segment)
        max_top_x = max(x + length for x, (length, _) in horizontal_segment['top'].items())
        max_bottom_x = max(x + length for x, (length, _) in horizontal_segment['bottom'].items())
        max_x = min(max_top_x , max_bottom_x) - offset
        
        self.normalizer_x = {'min': float(min_x), 'max': float(max_x)}
        
        # Keep the y coordinate normalization as is
        coords = np.array([data['pos'] for _, data in pedestrian_networkx_graph.nodes(data=True)])
        y_coords = coords[:, 1]
        self.normalizer_y = {'min': float(np.min(y_coords)), 'max': float(np.max(y_coords))}