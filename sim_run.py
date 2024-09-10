import math
import time 
import traci
import torch
import random
import gymnasium as gym
import numpy as np
import xml.etree.ElementTree as ET
from utils import convert_demand_to_scale_factor, scale_demand
from craver_config import (PHASES, DIRECTIONS_AND_EDGES, CONTROLLED_CROSSWALKS_DICT, initialize_lanes, get_phase_groups )

class CraverRoadEnv(gym.Env):
    def __init__(self, args, worker_id=None):
        super().__init__()
        self.vehicle_input_trips = args.vehicle_input_trips
        self.pedestrian_input_trips = args.pedestrian_input_trips

        # Modify file paths to include the unique suffix. Each worker has their own environment and hence their own copy of the trips file.
        self.worker_id = worker_id
        self.unique_suffix = f"_{worker_id}" if worker_id is not None else ""
        self.vehicle_output_trips = args.vehicle_output_trips.replace('.xml', f'{self.unique_suffix}.xml')
        self.pedestrian_output_trips = args.pedestrian_output_trips.replace('.xml', f'{self.unique_suffix}.xml')
        self.original_net_file = './SUMO_files/original_craver_road.net.xml'

        if args.manual_demand_veh is not None :
            # Convert the demand to scaling factor first
            scaling = convert_demand_to_scale_factor(args.manual_demand_veh, "vehicle", self.vehicle_input_trips)
            scale_demand(self.vehicle_input_trips, self.vehicle_output_trips, scaling, demand_type="vehicle")

        if args.manual_demand_ped is not None:
            # Convert the demand to scaling factor first
            scaling = convert_demand_to_scale_factor(args.manual_demand_ped, "pedestrian", self.pedestrian_input_trips)
            scale_demand(self.pedestrian_input_trips, self.pedestrian_output_trips, scaling, demand_type="pedestrian")

        self.use_gui = args.gui
        self.step_length = args.step_length
        self.max_timesteps = args.max_timesteps
        self.sumo_running = False
        self.step_count = 0
        self.auto_start = args.auto_start
        self.tl_ids = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # Only control this one for now
        
        self.previous_tl_action = None
        self.action_duration = args.action_duration  # Duration of each action in seconds
        # Number of simulation steps that should occur for each action. trying to ajuust for any given step length
        self.steps_per_action = int(self.action_duration / self.step_length) # This is also the size of the observation buffer
        print(f"Steps per action: {self.steps_per_action}")
        self.current_action_step = 0 # To track where we are within the curret action's duration

        self.phases = PHASES
        self.phase_groups = get_phase_groups(self.action_duration)
        self.controlled_crosswalks_dict = CONTROLLED_CROSSWALKS_DICT
        self.direction_and_edges = DIRECTIONS_AND_EDGES

        self.tl_lane_dict = {}
        self.tl_lane_dict['cluster_172228464_482708521_9687148201_9687148202_#5more'] = initialize_lanes()
        self.tl_pedestrian_status = {} # For pedestrians related to crosswalks attached to TLS.

        self.demand_scale_min = args.demand_scale_min
        self.demand_scale_max = args.demand_scale_max

        self.current_crosswalk_selection = None 
        self.current_phase_group = None
        self.current_state_index = None 
        self.corrected_occupancy_map = None

        # Create a bunch of reverse lookup dictionaries which can be referenced in various ways
        self.crosswalk_to_vicinity_walking_edges = {
            crosswalk_id: data['vicinity_walking_edges']
            for _, data in self.controlled_crosswalks_dict.items()
            for crosswalk_id in data['ids']
        }

        self.crosswalk_to_related_junction_edges = {
            crosswalk_id: data['related_junction_edges']
            for _, data in self.controlled_crosswalks_dict.items()
            for crosswalk_id in data['ids']
        }

        # Given any edge among all edges (including the corsswalk itself), return the re-route edge dict
        self.crosswalk_to_reroute_edges = {}
        for _, data in self.controlled_crosswalks_dict.items():
            reroute_edges = data['reroute_edges']
            all_edges = data['vicinity_walking_edges'] + data['connected_edges'] + data['ids']
            
            for edge in all_edges:
                self.crosswalk_to_reroute_edges[edge] = reroute_edges

        # Get the direction of the any edge around the crosswalk
        self.edge_to_direction = {}
        for direction, edges in self.direction_and_edges.items():
            for edge in edges:
                self.edge_to_direction[edge] = direction

        # Get the numerical id of the crosswalk, given any edge including the ID
        self.edge_to_numerical_crosswalk_id = {}
        for number, data in self.controlled_crosswalks_dict.items():
            all_edges = data['vicinity_walking_edges'] + data['connected_edges'] + data['ids']
            for edge in all_edges:
                self.edge_to_numerical_crosswalk_id[edge] = number 

        # Get the ids of all crosswalks. This includes 1 and 2 as well.
        self.all_crosswalk_ids = [crosswalk_id for _, data in self.controlled_crosswalks_dict.items() for crosswalk_id in data['ids']]
        self.controlled_crosswalk_mask = [0, 3, 4, 5, 6, 7, 8, 9, 10] # The crosswalks that can be disabled. 1 and 2 are not controlled. 
        self.controlled_crosswalks_masked_dict = {k: self.controlled_crosswalks_dict[k] for k in self.controlled_crosswalk_mask if k in self.controlled_crosswalks_dict}
        self.controlled_crosswalk_masked_ids = [crosswalk_id for _, data in self.controlled_crosswalks_masked_dict.items() for crosswalk_id in data['ids']]
        # Do these lists that are gotten from dict always maintain the same order (Yes/ No)? Answer: Yes, they do.
        #print(f"\n\nControlled crosswalk masked ids: {self.controlled_crosswalk_masked_ids}\n\n")

        self.pressure_dict = {tl_id: {'vehicle': {}, 'pedestrian': {}} for tl_id in self.tl_ids}
        self.pressure_dict['crosswalks'] = {c: 0 for c in self.controlled_crosswalk_masked_ids}

        self.directions = ['north', 'east', 'south', 'west']
        self.turns = ['straight', 'right', 'left']
        # The multiplier 4 is incoming, outgoing, inside, rerouted. The -1 because the crosswalk with 500 and 501 are the same. 
        self.single_obs_shape = (len(self.tl_ids)*(2 + 12 + 12 + 4 + 4 + 4) + (len(self.controlled_crosswalk_masked_ids) - 1)*4, ) # TL related (both veh and ped) + crosswalks related (only ped). 

        # For crosswalk control 
        self.walking_edges_to_reroute_from = []
        self.related_junction_edges_to_lookup_from = []
        self.alternative_crosswalks_flat = []
        self.currently_rerouted = []
        self.alternative_crosswalks_num = []
        self.crosswalks_to_disable = []

    def _get_vehicle_direction(self, signal_state):
        """
        """
        # Define signal bits for left and right blinkers
        VEH_SIGNAL_BLINKER_RIGHT = 0b1  # Bit 0
        VEH_SIGNAL_BLINKER_LEFT = 0b10  # Bit 1

        # Check if left blinker or right blinker is on
        left_blinker = bool(signal_state & VEH_SIGNAL_BLINKER_LEFT)
        right_blinker = bool(signal_state & VEH_SIGNAL_BLINKER_RIGHT)

        if left_blinker and not right_blinker:
            return "left"
        elif right_blinker and not left_blinker:
            return "right"
        else:
            # This covers cases where both blinkers are on (emergency) or off
            return "center"

    def _step_operations(self, occupancy_map, print_map=False, cutoff_distance=100):
        """
        Requires occupancy map as input. The changes made here should be reflected in the next time step's occupancy map.
        Some corrections have to be done every step.
        1. Update the pedestrian status when they cross: For each traffic light, check the outgoing pedestrians.If a pedestrian is in the outgoing area and hasn't been marked as 'crossed', update their status to 'crossed' in the self.tl_pedestrian_status dictionary.
        2. In case the same lanes are used for L, R, S turns (in case of vehicles and incoming). The straight lane will have repeated entries, remove them.  
        3. Vehicles are only included in the occupancy map if they are close to a given distance. In both incoming and outgoing directions.
        """
        # Handle outgoing pedestrians
        for tl_id in self.tl_ids:
            for _, persons in occupancy_map[tl_id]['pedestrian']['outgoing'].items():
                for person in persons:
                    if person not in self.tl_pedestrian_status or self.tl_pedestrian_status[person] != 'crossed':
                        # If the pedestrian crossed once, consider them as crossed (assume they wont cross twice, there is no way to know this without looking into their route, which is not practical.) 
                        self.tl_pedestrian_status[person] = 'crossed'

        # Handle special case for incoming vehicles
        for tl_id in self.tl_ids:
            for lane_group, vehicles in occupancy_map[tl_id]['vehicle']['incoming'].items():
                if lane_group not in ['south-straight', 'west-straight', 'east-straight', 'north-straight']:
                    ew_ns_direction = lane_group.split('-')[0]
                    straight_lane_group = f"{ew_ns_direction}-straight"
                    
                    # If this vehicle (which is in a non-straight lane) is also in the straight lane, remove it from the straight lane.
                    for vehicle in vehicles:
                        if vehicle in occupancy_map[tl_id]['vehicle']['incoming'][straight_lane_group]:
                            occupancy_map[tl_id]['vehicle']['incoming'][straight_lane_group].remove(vehicle) # Remove from the straight lane group
        
        # Handle vehicles
        for direction in ['incoming', 'outgoing']:
            for lane_group, vehicles in occupancy_map[tl_id]['vehicle'][direction].items():
                vehicles_to_remove = []
                for vehicle in vehicles:
                    distance = self._get_vehicle_distance_to_junction(tl_id, vehicle)
                    if distance > cutoff_distance:
                        vehicles_to_remove.append(vehicle)
                    
                # Remove vehicles outside the cutoff distance
                for vehicle in vehicles_to_remove:
                    occupancy_map[tl_id]['vehicle'][direction][lane_group].remove(vehicle)

        if print_map: # Corrected map
            print("\nOccupancy Map:")
            for id, data in occupancy_map.items():
                if id == 'crosswalks':
                    print(f"\nCrosswalks:")
                    for crosswalk_id, crosswalk_data in data.items():
                        print(f"    Crosswalk: {crosswalk_id}")
                        for direction, count in crosswalk_data.items():
                            print(f"        {direction.capitalize()}: {count}")
                else:
                    print(f"\nTraffic Light: {id}")
                    for agent_type in ["vehicle", "pedestrian"]:
                        print(f"  {agent_type.capitalize()}s:")
                        for direction in occupancy_map[id][agent_type].keys():
                            print(f"    {direction.capitalize()}:")
                            for lane_group, ids in data[agent_type][direction].items():
                                print(f"      {lane_group}: {len(ids)} [{', '.join(ids)}]")
                                if agent_type == "vehicle":
                                    for idx in ids:
                                        distance = self._get_vehicle_distance_to_junction(id, idx)
                                        print(f"        {idx}: {distance:.2f}m")   
            
        return occupancy_map

    def _get_vehicle_distance_to_junction(self, junction_id, vehicle_id):
        """
        Calculate the distance between a vehicle and a specific junction.

        :param junction_id: ID of the junction
        :param vehicle_id: ID of the vehicle
        :return: Distance between the vehicle and the junction in meters
        """
        try:
            # Get the x, y coordinates of the junction
            junction_pos = traci.junction.getPosition(junction_id)

            # Get the x, y coordinates of the vehicle
            vehicle_pos = traci.vehicle.getPosition(vehicle_id)

            # Calculate the Euclidean distance
            distance = math.sqrt(
                (junction_pos[0] - vehicle_pos[0])**2 + 
                (junction_pos[1] - vehicle_pos[1])**2
            )

            return distance

        except traci.TraCIException as e:
            print(f"Error calculating distance: {e}")
            return None
    
    def _update_pressure_dict(self, corrected_occupancy_map):
        """
        Update the data structure that holds info about pressure in outgoing directions.
        For both vehicles and pedestrians.

        For crosswalks, If the pedestrians are being rerouted, that means there is pressure that is not being addressed.
        Pressure = incoming (upside + downside) - outgoing (inside)
        However, if rerouted, then Pressure = -ve (rerouted)
        """

        for tl_id in self.tl_ids:
            #### VEHICLES ####

            # Initialize pressure and calculate for each direction
            vehicle_pressure = {d: 0 for d in self.directions}

            for outgoing_direction in self.directions:
                # Calculate incoming traffic towards this direction
                incoming = 0
                if outgoing_direction == 'north': # These four are outgoing directions
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['south-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['east-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['west-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['south-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['east-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['west-left'])

                elif outgoing_direction == 'south': # These four are outgoing directions
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['north-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['east-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['west-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['north-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['east-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['west-right'])

                elif outgoing_direction == 'east': # These four are outgoing directions
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['west-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['north-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['south-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['west-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['north-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['south-right'])

                elif outgoing_direction == 'west': # These four are outgoing directions
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['east-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['north-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['south-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['east-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['north-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['south-left'])
                
                # Calculate actual outgoing traffic
                outgoing = len(corrected_occupancy_map[tl_id]['vehicle']['outgoing'][outgoing_direction])
                
                # Calculate pressure
                vehicle_pressure[outgoing_direction] = incoming - outgoing
                self.pressure_dict[tl_id]['vehicle'][outgoing_direction] = vehicle_pressure[outgoing_direction]

            #### PEDESTRIANS (In this TL crossings) ####
            pedestrian_pressure = {d: 0 for d in self.directions}

            for outgoing_direction in self.directions:
                # Calculate incoming pedestrians towards this direction
                incoming = len(corrected_occupancy_map[tl_id]['pedestrian']['incoming'][outgoing_direction])
                
                # Calculate actual outgoing pedestrians
                outgoing = len(corrected_occupancy_map[tl_id]['pedestrian']['outgoing'][outgoing_direction])
                
                # Calculate pressure
                pedestrian_pressure[outgoing_direction] = incoming - outgoing
                self.pressure_dict[tl_id]['pedestrian'][outgoing_direction] = pedestrian_pressure[outgoing_direction]

        #### CROSSWALKS ####  
        for crosswalk_id in self.controlled_crosswalk_masked_ids:
            if crosswalk_id != ':9687187501_c0': # Special case crosswalk
                incoming = len(corrected_occupancy_map['crosswalks'][crosswalk_id]['upside']) + len(corrected_occupancy_map['crosswalks'][crosswalk_id]['downside'])
                outgoing = len(corrected_occupancy_map['crosswalks'][crosswalk_id]['inside'])
                rerouted = len(corrected_occupancy_map['crosswalks'][crosswalk_id]['rerouted'])

                if rerouted > 0:
                    self.pressure_dict['crosswalks'][crosswalk_id] = -rerouted
                else:
                    self.pressure_dict['crosswalks'][crosswalk_id] = incoming - outgoing

    def _get_occupancy_map(self, ):
        """
        Features: 
            - If the same lane is used for multiple directions, the indicator light of vehicle is used to determine the direction. (The indicator light turns on about 100m far from the junction.)
        """

        occupancy_map = {}

        # For the traffic light and its related components
        for tl_id, lanes in self.tl_lane_dict.items():
            occupancy_map[tl_id] = {
                "vehicle": {
                    "incoming": {},
                    "inside": {}, # Inside the junction
                    "outgoing": {}
                },
                "pedestrian": {
                    "incoming": {},
                    "outgoing": {}
                }
            }
            
            for agent_type in ["vehicle", "pedestrian"]:
                for direction in  occupancy_map[tl_id][agent_type].keys():
                    for lane_group, lane_list in lanes[agent_type][direction].items():
                        occupancy_map[tl_id][agent_type][direction][lane_group] = []
                        for lane in lane_list:
                            if agent_type == "vehicle":

                                if lane != '-1':  # Skip lanes that are common for all directions
                                    ids = traci.lane.getLastStepVehicleIDs(lane) if "edge" not in lane else traci.edge.getLastStepVehicleIDs(lane.split('.')[1]) # Its actually an edge in the else case.
                                    occupancy_map[tl_id][agent_type][direction][lane_group].extend(ids)
                                else: 
                                    # If there are multiple -1s, this case can occur multiple times.
                                    # In this case, look at the indicator light of the vehicle to get the direction.
                                    # Get the EW-NS direction and the current turn direction, then all vehicles in the straight lane group
                                    ew_ns_direction = lane_group.split('-')[0]
                                    turn_direction = lane_group.split('-')[1]

                                    straight_lane_group = f"{ew_ns_direction}-straight"

                                    # TODO: If there are multiple straight lanes where vehicles that want to go left or right also exist, then need to account for that
                                    straight_lane = lanes[agent_type][direction][straight_lane_group][0] 
                                    existing_ids = traci.lane.getLastStepVehicleIDs(straight_lane) if "edge" not in lane else traci.edge.getLastStepVehicleIDs(lane.split('.')[1])  # Its actually an edge in the else case.

                                    #print(f"Straight lane: {straight_lane}, Existing ids: {existing_ids}")

                                    if len(existing_ids)>0:
                                        #print(f"Vehicle exists")
                                        new_ids = []
                                        for veh_id in existing_ids:
                                            signal_state = traci.vehicle.getSignals(veh_id)
                                            veh_direction = self._get_vehicle_direction(signal_state)
                                            # print(f"Vehicle: {veh_id}, Signal: {signal_state}, Direction: {veh_direction}")
                                            if veh_direction == turn_direction:
                                                new_ids.append(veh_id)

                                        occupancy_map[tl_id][agent_type][direction][lane_group].extend(new_ids) 

                            else:  # pedestrian
                                if lane.startswith(':'):  # Check if it's an internal lane
                                    # Doing it the computationally expensive way
                                    # Get all persons in the simulation
                                    all_persons = traci.person.getIDList()
                                    # Filter persons on this junction
                                    for person in all_persons:
                                        if traci.person.getRoadID(person) == lane:
                                            # If not crossed yet, add to incoming 
                                            
                                            if direction == "incoming":
                                                if person not in self.tl_pedestrian_status or self.tl_pedestrian_status[person] != 'crossed': 
                                                    occupancy_map[tl_id][agent_type][direction][lane_group].append(person)
                                            else: 
                                                # Add to outgoing, just being inside the crossing is enough.
                                                occupancy_map[tl_id][agent_type][direction][lane_group].append(person)

                                else: 
                                    print("Only implemented to work with JunctionDomain. Not implemented yet for external lanes or edges")

        # For the crosswalks related components
        # For each crosswalk, get the occupancy in the upside, downside, inside, and rerouted.
        # Do we get the pedestrian ids or do we just get the count? # For now, let's get the ids.
        # If they are being-rerouted, they should not be a part of upside, downside, and inside. 
        occupancy_map['crosswalks'] = {}
        for crosswalk_id in self.controlled_crosswalk_masked_ids:
            occupancy_map['crosswalks'][crosswalk_id] = {
                "upside": [],
                "downside": [],
                "inside": [],
                "rerouted": [] }

            # These already contain internal edges
            vicinity_walking_edges = self.crosswalk_to_vicinity_walking_edges[crosswalk_id]
        
            # For upside and downside
            for edge in vicinity_walking_edges:
                pedestrians = traci.edge.getLastStepPersonIDs(edge)
                direction = self.edge_to_direction[edge]
                occupancy_map['crosswalks'][crosswalk_id][direction].extend(pedestrians) # Incase ids are wanted, use the list instead of the length

            # For inside, use the crosswalk id itself.
            pedestrians = traci.edge.getLastStepPersonIDs(crosswalk_id)
            occupancy_map['crosswalks'][crosswalk_id]['inside'].extend(pedestrians)

            # Add re-routed pedestrians
            # If this crosswalk happens to be disabled, then add the upside and downside values to get the rerouted value. Setting upside, downside to 0.
            # Inside may contain pessengers that are in the process of crossing when the new decision is made. Not setting that to 0.
            if crosswalk_id in self.crosswalks_to_disable:
                occupancy_map['crosswalks'][crosswalk_id]['rerouted'] = occupancy_map['crosswalks'][crosswalk_id]['upside'] + occupancy_map['crosswalks'][crosswalk_id]['downside']
                occupancy_map['crosswalks'][crosswalk_id]['upside'] = []
                occupancy_map['crosswalks'][crosswalk_id]['downside'] = []

        # Special case: This id 'ids': [':9687187500_c0', ':9687187501_c0'] represents a single disjoint crosswalk. Do not repeat the counts twice.
        # Just use the 9687187500_c0 once and skip the second part of the special case crosswalk (only for upside and downside) because it would have been counted in the first part.
        # This is the special case crosswalk # Since this is updating for inside (it is not affected by re-routing)
        # Also not affected by other possible problems beacuse of the order [500 comes before 501 inthe crosswalks list]
        pedestrians = traci.edge.getLastStepPersonIDs(crosswalk_id)
        occupancy_map['crosswalks'][':9687187500_c0']['inside'].extend(occupancy_map['crosswalks'][':9687187501_c0']['inside']) # Use ':9687187500_c0'
        del occupancy_map['crosswalks'][':9687187501_c0'] # Delete the second part of the special case crosswalk

        # for crosswalk_id in self.controlled_crosswalk_masked_ids:
        #     if crosswalk_id != ':9687187501_c0':
        #         print(f"\nStep:{self.step_count}\n\nCrosswalk: {crosswalk_id}\nUpside: {occupancy_map['crosswalks'][crosswalk_id]['upside']}\nDownside: {occupancy_map['crosswalks'][crosswalk_id]['downside']}\nInside: {occupancy_map['crosswalks'][crosswalk_id]['inside']}\nRerouted: {occupancy_map['crosswalks'][crosswalk_id]['rerouted']}")
        return occupancy_map
    
    @property
    def action_space(self):
        """
        In the simplified action space with phase groups, the agents decision is binary.
        Changed this to use MultiDiscrete 
        """
        num_actions = len(self.tl_ids) # Number of traffic lights where the choice is between the number of phase groups (in this case just two).
        num_actions += len(self.controlled_crosswalk_mask) # Plus the size of the controlled_crosswalks_masked_dict (enable/ disable)
        return gym.spaces.MultiDiscrete([2] * num_actions) # 2 indicates the binary choice
        
    @property
    def observation_space(self):
        """
        Each timestep (not action step) observation is the pressure in all outgoing directions.
        TODO: We do not have lane level granularity yet (in both action and observation).
        """
        # The observation is the entire observation buffer
        return gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(self.steps_per_action, int(self.single_obs_shape[0])),
            dtype=np.float32
        )

    def step(self, action):
        """
        """
        if not self.sumo_running:
            raise Exception("Environment is not running. Call reset() to start the environment.")
        
        reward = 0
        done = False
        observation_buffer = []
        current_tl_action = action[0].item() # Convert tensor to int

        # Run simulation steps for the duration of the action
        for _ in range(self.steps_per_action):
            
            # Apply action needs to happen every timestep
            self._apply_action(action, self.current_action_step, self.previous_tl_action)

            traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
            self.step_count += 1
            # Increment the current action step
            self.current_action_step = (self.current_action_step + 1) % self.steps_per_action # Wrapped around some modulo arithmetic

            # Collect observation at each substep
            obs = self._get_observation(print_map=False)
            #print(f"\nObservation: {obs}")
            observation_buffer.append(obs)

            self._update_pressure_dict(self.corrected_occupancy_map)

            # Accumulate reward
            reward += self._get_reward(current_tl_action)
            
            # Check if episode is done
            if self._check_done():
                done = True
                break

        # formatted_buffer = "\n".join(f"{arr})" for arr in observation_buffer)
        # print(f"\nAccumulated Observation:\n{formatted_buffer}")
        # print(f"\nCurrent Action: {action}")
        #print(f"\nAccumulated Reward: {reward}")
        self.previous_tl_action = current_tl_action
        observation = np.asarray(observation_buffer) # shape (steps_per_action, 74); e.g. (10, 74) with 10 items each of size 74 
        #print(f"\nAccumulated Observation:\n{observation}, shape: {observation.shape}")
        info = {}

        return observation, reward, done, False, info
        
    def _get_observation(self, print_map=False):
        """
        This is per step observation.
        About including previous action in the observation:
            - Each action persists for a number of timesteps and the observation is collected at each timestep.
            - Therefore, the previous action is the same for a number of timesteps. This adds too much extra computational overhead for the model. 
            - It would have been fine if model was MLP (we can just attach the previous action at the end)
            But for CNN, it breaks the grid structure.
        Pressure itself is not a part of the observation. It is only used for reward calculation.
        """
        
        # Get the occupancy map and print it
        occupancy_map = self._get_occupancy_map()
        self.corrected_occupancy_map = self._step_operations(occupancy_map, print_map=print_map, cutoff_distance=100)
        
        observation = []
        # Traffic lights
        for tl_id in self.tl_ids:
            #### Current phase group info (This changes even within the action timesteps) ####
            current_phase_group = []
            current_phase_group.append(self.current_phase_group)
            current_phase_group.append(self.current_state_index/1) # To prevent a value of 2 [0, 1, 2] -> [0, 0.5, 1]
            observation.extend(current_phase_group)

            #### VEHICLES INFO ####
            # Incoming
            for outgoing_direction in self.directions:
                for turn in self.turns:
                    incoming = len(self.corrected_occupancy_map[tl_id]['vehicle']['incoming'][f"{outgoing_direction}-{turn}"])
                    observation.append(incoming)

            # Inside
            for outgoing_direction in self.directions:
                for turn in self.turns:
                    inside = len(self.corrected_occupancy_map[tl_id]['vehicle']['inside'][f"{outgoing_direction}-{turn}"])
                    observation.append(inside)

            # Outgoing
            for outgoing_direction in self.directions:
                outgoing = len(self.corrected_occupancy_map[tl_id]['vehicle']['outgoing'][outgoing_direction])
                observation.append(outgoing)
                
            #### PEDESTRIANS INFO ####
            # Incoming
            for outgoing_direction in self.directions:
                incoming = len(self.corrected_occupancy_map[tl_id]['pedestrian']['incoming'][outgoing_direction])
                observation.append(incoming)

            # Outgoing
            for outgoing_direction in self.directions:
                outgoing = len(self.corrected_occupancy_map[tl_id]['pedestrian']['outgoing'][outgoing_direction])
                observation.append(outgoing)

        observation = np.asarray(observation, dtype=np.float32)
        # Create a mask for part of observations that do not need normalization i.e., a bunch of elements at the beginning (which do not need a normalization)
        mask = np.tile([False] * (len(self.tl_ids)*len(current_phase_group)) + \
                       [True] * (len(observation) - len(self.tl_ids)*len(current_phase_group)), len(self.tl_ids)) # This seems overly complicated but isnt
        # Normalize the items other than the masked ones
        normalizer = 10.0 # TODO: A better normalization scheme?
        observation[mask] /= normalizer

        # Concatenate the crosswalk info (4 obs for each crosswalk)
        crosswalk_info = []
        for crosswalk_id in self.controlled_crosswalk_masked_ids: # ':9687187501_c0' and ':9687187500_c0' are same special case crosswalk
            if crosswalk_id != ':9687187501_c0': # This if condition is a huge tax to the system.
                crosswalk_info.append(len(self.corrected_occupancy_map['crosswalks'][crosswalk_id]['upside']))
                crosswalk_info.append(len(self.corrected_occupancy_map['crosswalks'][crosswalk_id]['downside']))
                crosswalk_info.append(len(self.corrected_occupancy_map['crosswalks'][crosswalk_id]['inside']))
                crosswalk_info.append(len(self.corrected_occupancy_map['crosswalks'][crosswalk_id]['rerouted']))

        # Normalize the crosswalk info
        crosswalk_info = np.asarray(crosswalk_info, dtype=np.float32)
        crosswalk_info /= normalizer # TODO: A better normalization scheme? 
        observation = np.concatenate((observation, crosswalk_info))
        return observation
    
    def _apply_action(self, action, current_action_step, previous_tl_action=None):
        """
        In the action space with phase groups, previous action is used to determine if there was a switch.
        Later, actions were added to whether or not enable the crosswalks.

        previous_action will be None none in reset.
        apply_action is called every step.
        """
        #print(f"\nCurrent Action: {action}, TL action: {action[0]} Previous TL Action: {previous_tl_action}")

        # For the TL control. For the same action, what state is applied changes within the action timesptes
        current_tl_action = action[0].item() # Convert tensor to int
        self.current_phase_group = current_tl_action # Convert tensor to int
        # For action space with phase groups
        if previous_tl_action == None: # First action 
            previous_tl_action = current_tl_action # Assume that there was no switch

        if current_tl_action != previous_tl_action:
            # print("Switching phase group")
            # Switch the phase group
            for tl_id in self.tl_ids:
                
                # All these shenanigans is to determine the index. TODO: Is there an efficient way to do this?
                durations = [phase["duration"] for phase in self.phase_groups[current_tl_action]]
                cumulative_durations = [sum(durations[:i+1]) for i in range(len(durations))] # [4, 5, 10]
                for i, duration in enumerate(cumulative_durations):
                    if current_action_step < duration:
                        index = i
                        break

                self.current_state_index = index
                state = self.phase_groups[current_tl_action][index]["state"]
                # print(f"Setting phase: {state}")
                traci.trafficlight.setRedYellowGreenState(tl_id, state)
                    
        else: # No switch. Just continue with the green in this phase group.
            # print("Continuing with the same phase group")
            self.current_state_index = 2
            for tl_id in self.tl_ids:
                state = self.phase_groups[current_tl_action][2]["state"] # Index is always 2
                # print(f"Setting phase: {state}")
                # Skip the first two phases, they are for buffering the transition.
                traci.trafficlight.setRedYellowGreenState(tl_id, state)

        # For the same action, what choice is performed remains same within the action timesteps
        # Only update the corsswalk control lists when current_action_step is 0 i.e., when a new action is gotten
        if current_action_step == 0:   
            # Clear the previous values 
            self.walking_edges_to_reroute_from = []
            self.related_junction_edges_to_lookup_from = []
            self.alternative_crosswalks_flat = []
            self.currently_rerouted = []
            self.crosswalks_to_disable = []
            self.alternative_crosswalks_num = []
            
            current_crosswalk_action = action[1:].tolist() # The first element is the TL action
            # controlled_crosswalks_dict goes from 0 to 10, but we need to exclude 1 and 2
            for i in range(len(current_crosswalk_action)): 
                if current_crosswalk_action[i] == 0: # 0 means disable 

                    j = self.controlled_crosswalk_mask[i] # Get the actual crosswalk number (key), excludes 1 and 2
                    crosswalks_list = self.controlled_crosswalks_masked_dict[j]['ids'] # Get the crosswalks that are controlled by this action
                    for item in crosswalks_list: # Need a for loop because of them contains multiple crosswalks in a list
                        self.crosswalks_to_disable.append(item)

            ######################################
            # For testing purposes.
            # Based on the timesteps, allow a bunch and then disallow the bunch
            # print(f"Step count: {self.step_count}") # Increments by 10 here
            
            # # Define time ranges for disabling crosswalks
            # time_ranges = [
            #     (200, 1000, self.all_crosswalk_ids[3:7]),
            # ]

            # # Determine which crosswalks to disable based on current step
            # self.crosswalks_to_disable = [] 
            # for start, end, crosswalks in time_ranges:
            #     if start < self.step_count <= end:
            #         #print(f"\nTime range: {start} - {end}, Disabled Crosswalks: {crosswalks}")
            #         self.crosswalks_to_disable = crosswalks
            #         break
            
            # End of testing purposes code
            ######################################

            # self.all_crosswalks contains 1 and 2, controlled_crosswalks_masked_dict does not
            #print(f"All crosswalks: {self.controlled_crosswalk_masked_ids}\nCrosswalks to disable: {self.crosswalks_to_disable}")

            for crosswalk_id in self.crosswalks_to_disable:
                edges = self.crosswalk_to_vicinity_walking_edges[crosswalk_id] # Does not contain the crosswalk itself, just the vicinity walking edges list
                self.walking_edges_to_reroute_from.extend(edges)

                lookup_edges = self.crosswalk_to_related_junction_edges[crosswalk_id] # Just the lookup edges i.e., related_junction_edges
                self.related_junction_edges_to_lookup_from.extend(lookup_edges)
                
            # Find alternative crosswalks ids (they should include crosswalk 1 and 2, because 1 and 2 are present in the sim. we are controlling them as part of TL) and flatten
            for crosswalk_data in self.controlled_crosswalks_dict.values():
                if not any(cw in self.crosswalks_to_disable for cw in crosswalk_data['ids']):
                    self.alternative_crosswalks_flat.extend(crosswalk_data['ids'])

            # Get the numerical crosswalk ids
            self.alternative_crosswalks_num = [self.edge_to_numerical_crosswalk_id[crosswalk_id] for crosswalk_id in self.alternative_crosswalks_flat]

        # Although the crosswalks to disable are gotten every 10 timesteps, the enforcement of the action (i.e., re-routing pedestrians) is done every step.
        # This is going to be severely computationally taxing on the simulation.
        self._disallow_pedestrians(self.walking_edges_to_reroute_from, self.related_junction_edges_to_lookup_from)
        #print(f"\nDisabled crosswalks: {disabled_crosswalks}\n")

    def _get_reward(self, current_tl_action):
        """
        - Consider both Vehicles and Pedestrians
        - Penalize frequent changes of action
        - Since pressure is calculated as incoming - outgoing. Pressure being high is not good.
        - To penalize high pressure, we make pressure negative.
        """

        reward = 0
        vehicle_pressure = 0
        pedestrian_pressure = 0
        crosswalks_pressure = 0

        # Related to Traffic Light (vehicle and pedestrian crossings)
        for tl_id in self.tl_ids:
            # Use stored pressures for reward calculation
            for agent_type in ['vehicle', 'pedestrian']:
                for direction in self.directions:
                    pressure = self.pressure_dict[tl_id][agent_type][direction]

                    if agent_type == 'vehicle':
                        vehicle_pressure += pressure
                    else:
                        pedestrian_pressure += pressure

        # Related to Pedestrian crosswalks (If the pedestrians are re-routed, pressure is negative else it is incoming (upside + downside) - outgoing (inside crosswalk))
        controlled_crosswalk_pressures = []
        for crosswalk_id in self.controlled_crosswalk_masked_ids:
            if crosswalk_id != '9687187501_c0':
                controlled_crosswalk_pressures.append(self.pressure_dict['crosswalks'][crosswalk_id]) 

        # Only collect the positive pressure values
        crosswalks_pressure = sum(pressure for pressure in controlled_crosswalk_pressures if pressure > 0)
                
        # Pressure is bad. So we make it negative. Get the values from wandb config.
        reward = -0.33*vehicle_pressure - 0.33*pedestrian_pressure - 0.33*crosswalks_pressure

        # Frequency penalty
        if self.previous_tl_action is not None and current_tl_action != self.previous_tl_action:
            reward -= 0.5  # Penalty for changing tl actions. Since this is per step reward. Action change is reflected multiplied by action steps.
        
        # Re-route penalty (because any re-route increases travel time). Only collect the negative pressure values
        reroute_pressure = sum(pressure for pressure in controlled_crosswalk_pressures if pressure < 0) # This is already negative.
        reward -= reroute_pressure

        #print(f"\nStep Reward: {reward}")
        return reward

    def _check_done(self):
        """
        TODO: What more conditions can be added here?
        - Gridlock? Jam? of vehicles or pedestrians? Crashes?
        """
        return self.step_count >= self.max_timesteps

    def _modify_net_file(self, crosswalks_to_disable):
        """
        Just for changing the appearence of disallowed crosswalks. Not used right now.
        """
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

    def _disallow_pedestrians(self, walking_edges_to_reroute_from, related_junction_edges_to_lookup_from):
        """ 
        Disallow pedestrians means reroute pedestrians from the nearest possible crosswalk.
        This is called once per action i.e., after 10 actual simulation steps. 
        This means some pedestrians which have spawned and reached the crosswalks in the last 10 steps will not be rerouted. Which is unlikely.

        One important consideration is: 
        1. When to check if pedestrians need a re-route: whereever they may be at the action time step (because we go through all pedestrians) 
        2. When to actually perform the re-route: If they are present in the vicinity of the crosswalk. i.e., only if they are nearby, they will be able to see that a crosswalk is disabled. (closer to real-world scenario) 

        # We cannot check if a pedestrian far away has a crosswalk in their route (which they reach sometime in the future) and then re-route them immediately.
        # If the pedestrian in already in the crosswalk itself at the time of action, they dont need to be re-routed.

        Pedestrian current edge can be the crosswalk id itself.. or the walking areas associated. Both begin with :, Both will be invisible in the route but can be obtained by current edge.  
        Pedestrian can be in the internal edge (related to the junction) which will be _w0 or _w1 attached to the junction name. This can also be obtained using current edge.
        # Done.
         10. Only route if they have already not been re-routed or something like that?
         3. Why are the re-routed pedestrians actually not going to the other side of crosswalk?
         4. Measure the efficacy of the re-routing.
         5. Convert to shortest path method.
         8. Track the edges from which pedestrians are being missed.
         3. Since the re-routing happens every 10 timesteps, that can be too late. Because pedestrians keep moving beween the two decision times. Switched to enforecement of disabling crosswalks every step.
         2. Pedestrians who are just passing by, not too close, should not be re-routed.
         6. One step forward lookup is necessary because we can only get the last step pedestrian info from the simulation. For forward lookup, we have their remaining route, and we have the related junction edges, make use of that.
         7. Make the second route where they go to the other side of the crosswalk. The third one to the destination.
         1. After assigning the new route, move the pedestrian to the first edge/ lane of the new route. (teleport)? The teleportation from simulation seems to work fine. And my own teleport does not
        """

        for ped_id in traci.person.getIDList():
            if ped_id not in self.currently_rerouted: # If they are already re-routed, no need to re-route them again. Until the next action where this list gets reset.
                current_edge = traci.person.getRoadID(ped_id) # This may contain an internal edge.
                
                # Get the remaining edges in the person's route
                remaining_stages_count = traci.person.getRemainingStages(ped_id)
                remaining_edges = []
                for i in range(remaining_stages_count):  
                    stage = traci.person.getStage(ped_id, i)
                    remaining_edges.extend(stage.edges)
                
                # For all pedestrians, print their route
                # print(f"\nPedestrian {ped_id} remaining route: {remaining_edges}\n current edge: {current_edge}\n")
                # print(f"\nWalking edges to reroute from: {walking_edges_to_reroute_from}\n")
                # print(f"\nRelated junction edges to lookup from: {related_junction_edges_to_lookup_from}\n")

                # If the person is in the vicinity of the crosswalk we want to disable or the look forward in the current edge by 1.
                # If the person is directly on the edge that we want to disable, then they are continued to walk
                # Disabling the forward lookup. Works without problems.
                # If forward lookup is enabled, the auto-teleportation is too harsh. We need to do controlled teleportation 
                if self._check_vicinity(current_edge, walking_edges_to_reroute_from, remaining_edges, related_junction_edges_to_lookup_from, forward_lookup= False): # If the route includes the ones we want to disable.
                    
                    # Get the destination (end) edge
                    destination_edge = remaining_edges[-1] # Last edge of the last stage is the destination.

                    # Based on whether current edge is upside or downside, select the new crosswalk's downside or upside.
                    current_direction = self.edge_to_direction.get(current_edge) # This is the direction of the current edge.
                    other_direction = 'upside' if current_direction == 'downside' else 'downside' # Just a simple way to get the other direction.

                    # Choice of which alternate crosswalk to choose is based on shortest path. 
                    # Among the alternate crosswalks, for each pedestrian, find the closest crosswalk.
                    current_crosswalk_num = self.edge_to_numerical_crosswalk_id.get(current_edge)
                    # make use of self.alternative_crosswalks_num to calculate smallest difference with current_crosswalk_num
                    differences = [abs(current_crosswalk_num - crosswalk_num) for crosswalk_num in self.alternative_crosswalks_num]
                    closest_crosswalk_index = differences.index(min(differences))
                    new_crosswalk_num = self.alternative_crosswalks_num[closest_crosswalk_index]

                    # This has to be gotten from the unmasked one because we need to include 1 and 2
                    new_crosswalk_id = self.controlled_crosswalks_dict[new_crosswalk_num]['ids'][0] # Just get the first one.
                    
                    # print(f"\nPedestrian {ped_id} is being re-routed from crosswalk {current_crosswalk_num} to crosswalk {new_crosswalk_num} with ID: {new_crosswalk_id}")
                    # print(f"Alternate crosswalk nums: {self.alternative_crosswalks_num}, differences: {differences}\n")

                    # Get the re-route point related to this new crosswalk
                    next_reroute_edge = self.crosswalk_to_reroute_edges[new_crosswalk_id].get(current_direction) # Understand the difference between teleport point and reroute point.

                    # Append two new walking stages:
                    # Althrough the routing can find a route from current edge directly to the destination edge, this is a problem because it can repeat the same route. 
                    # Moreoever, we want to ensure that we pass through an enabled crosswalk. Hence, we do routing in two stages.
                    # #    - One from the current edge to the new crosswalk 
                    found_route = traci.simulation.findIntermodalRoute(current_edge, next_reroute_edge, modes='') # Walking is the default mode. This returns a Stage object.
                    #print(f"\nFound route: {found_route}\n")

                    other_side_of_crosswalk = self.crosswalk_to_reroute_edges[new_crosswalk_id].get(other_direction)
                    found_route_2 = traci.simulation.findIntermodalRoute(next_reroute_edge, other_side_of_crosswalk, modes='') 
                    #print(f"\nFound route 2: {found_route_2}\n")

                    #   - Other from the new crosswalk to the destination edge
                    found_route_3 = traci.simulation.findIntermodalRoute(other_side_of_crosswalk, destination_edge, modes='') # Walking is the default mode. This returns a Stage object.
                    #print(f"\nFound route 3: {found_route_2}\n")

                    # Clear all the remaining stages of the pedestrian
                    # If a new stage is not immediately appended, this automatically removes the person in the next timestep.
                    traci.person.removeStages(ped_id)
                    
                    # Since we are finding intermodal route, this route could potentially have had many stages. Just append the first one. which is walking to the destination.
                    traci.person.appendStage(ped_id, found_route[0]) 
                    traci.person.appendStage(ped_id, found_route_2[0]) 
                    traci.person.appendStage(ped_id, found_route_3[0])

                    # If they got re-routed, change color to red
                    traci.person.setColor(ped_id, (255, 0, 0, 255))
                    self.currently_rerouted.append(ped_id)

    def _check_vicinity(self, current_edge, walking_edges_to_reroute_from, remaining_edges, related_junction_edges_to_lookup_from, forward_lookup=False):
        """
        If the current edge is already in the vicinity of the crosswalk to disable
        Or the next edge in the forward lookup in the route is in the vicinity
        For the fowrard step lookup, we need to know next edge. However, if cant even determine the current edge if they are internal.
        """
        # Basic conditon.
        if current_edge in walking_edges_to_reroute_from:
            #print(f"\nCurrent edge: {current_edge} is in the vicinity\n")
            return True
        else: 
            return False 
            # If foward lookup is disabled, comment this entire block for efficiency. 
            # IMPORTANT: If they are currently in the junction, and the next edge is among the one we want to diasble.
            # Forward lookup of one step does not work because, remaining_edges[0] does not give the exact remaining route but rather the first edge in entire route. 
            # If that first edge happens to be an internal edge, then it wont be present in the walking_edges_to_reroute_from.
            
            # if forward_lookup: # Forward lookup of the entire remaining route. 
            #     if current_edge in related_junction_edges_to_lookup_from and any(next_edge in walking_edges_to_reroute_from for next_edge in remaining_edges): # If any edge in the remaining route is in the vicinity.                         
            #         #print(f"\nNext edge in the future is in the vicinity\n")
            #         return True
            #     else: 
            #         print(f"\nCurrent edge: {current_edge} is not in the vicinity\n")
            #         return False
            # else:
            #     return False

    def reset(self, options=None):
        """
        """
        super().reset()
        if self.sumo_running:
            time.sleep(5) # Wait until the process really finishes 
            traci.close(False) #https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
        
        # Automatically scale demand (separately for pedestrian and vehicle)
        scale_factor_vehicle = random.uniform(self.demand_scale_min, self.demand_scale_max)
        scale_factor_pedestrian = random.uniform(self.demand_scale_min, self.demand_scale_max)

        scale_demand(self.vehicle_input_trips, self.vehicle_output_trips, scale_factor_vehicle, demand_type="vehicle")
        scale_demand(self.pedestrian_input_trips, self.pedestrian_output_trips, scale_factor_pedestrian, demand_type="pedestrian")

        # This should be done here before the SUMO call. This can disallow pedestrians before the simulation run.
        # Randomly select crosswalks to disable
        # to_disable = random.sample(self.controlled_crosswalks, min(5, len(self.controlled_crosswalks)))
        # Before sumo call 
        # self._modify_net_file(to_disable)

        if self.auto_start:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", 
                        "--verbose",
                        "--start" , 
                        "--quit-on-end", 
                        "-c", "./SUMO_files/craver.sumocfg", 
                        "--step-length", str(self.step_length),
                        "--route-files", f"{self.vehicle_output_trips},{self.pedestrian_output_trips}"
                        ]
                        
        else:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", 
                        "--verbose",
                        "--quit-on-end", 
                        "-c", "./SUMO_files/craver.sumocfg", 
                        "--step-length", str(self.step_length),
                        "--route-files", f"{self.vehicle_output_trips},{self.pedestrian_output_trips}"
                        ]
                        
        max_retries = 3
        try:
            for attempt in range(max_retries):
                try:
                    traci.start(sumo_cmd)
                    break
                except traci.exceptions.FatalTraCIError:
                    if attempt < max_retries - 1:
                        print(f"TraCI connection failed. Retrying... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(10)
                    else:
                        print(f"Failed to start TraCI after {max_retries} attempts.")
                        raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise

        # This should be done here after the SUMO call. As this can disallow pedestrians during the simulation run. 
        # Disallow pedestrians in some crosswalks. After sumo call beacuse we need traci.
        # disabled_crosswalks = self._disallow_pedestrians(to_disable)
        # print(f"\nDisabled crosswalks: {disabled_crosswalks}\n")

        self.sumo_running = True
        self.step_count = 0 # This counts the timesteps in an episode. Needs reset.
        self.current_action_step = 0
        self.tl_lane_dict = {}
        self.tl_lane_dict['cluster_172228464_482708521_9687148201_9687148202_#5more'] = initialize_lanes()

        # Randomly initialize the actions (current phase group and the current choice of crosswalks to activate) 
        self.current_phase_group = random.choice(list(self.phase_groups.keys()))
        self.current_crosswalk_selection = np.random.randint(2, size=len(self.controlled_crosswalks_masked_dict)).tolist() # 2 because of binary choice
        initial_action = torch.tensor([self.current_phase_group] + self.current_crosswalk_selection) # Make it a tensor so that its compatible with other outputs

        # Initialize the observation buffer
        observation_buffer = []
        for step in range(self.steps_per_action):
            # Apply the current phase group using _apply_action
            self._apply_action(initial_action, step, None)
            traci.simulationStep()
            obs = self._get_observation()
            observation_buffer.append(obs)

        observation = np.asarray(observation_buffer, dtype=np.float32)
        #print(f"\nInitial observation inside: {observation}\n")
        #print(f"\nInitial observation inside shape: {observation.shape}\n")
        info = {}
        return observation, info

    def close(self):
        if self.sumo_running:
            traci.close(False) #https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
            self.sumo_running = False