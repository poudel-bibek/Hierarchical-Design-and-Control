import math
import traci
import random
import gymnasium as gym
import numpy as np
from utils import convert_demand_to_scale_factor, scale_demand

class CraverRoadEnv(gym.Env):
    def __init__(self, args):
        super().__init__()
        """
        Phase information (22 characters). Incoming vehicle lanes, or pedestrian crossings
        1: east-right
        2: east-straight
        3: east-left
        4: east-uturn
        5: south-right
        6: south-straight
        7: south-left
        8: south-uturn
        9: west-right
        10: west-straight
        11: west-left
        12: west-uturn
        13: north-right
        14: north-straight
        15: north-left
        16: north-uturn
        17: pedestrian crossing north
        18: unused
        19: pedestrian crossing east (new)
        20: pedestrian crossing south
        21: unused
        22: pedestrian crossing west
        """

        if args.manual_demand_vehicles is not None :
            # Convert the demand to scaling factor first
            scaling = convert_demand_to_scale_factor(args.manual_demand_vehicles, type="vehicle")
            scale_demand(args.vehicle_input_trips, args.vehicle_output_trips, scaling, demand_type="vehicle")

        if args.manual_demand_pedestrians is not None:
            # Convert the demand to scaling factor first
            scaling = convert_demand_to_scale_factor(args.manual_demand_pedestrians, type="pedestrian")
            scale_demand(args.pedestrian_input_trips, args.pedestrian_output_trips, scaling, demand_type="pedestrian")

        self.use_gui = args.gui
        self.step_length = args.step_length
        self.max_timesteps = args.max_timesteps
        self.sumo_running = False
        self.step_count = 0
        self.auto_start = args.auto_start
        self.tl_ids = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # Only control this one for now
        
        # Original 10 phases defaulted by SUMO with 93 seconds cycle time
        # TODO: Correct this with the new link info. Done
        # Keep this, required for TL only evaluation
        self.phases = {'cluster_172228464_482708521_9687148201_9687148202_#5more': [
            {"duration": 32, "state": "rrrrgGggrrrrgGggrrGrrG"},
            {"duration": 5, "state": "rrrrgGggrrrrgGggrrrrrr"},
            {"duration": 4, "state": "rrrryyggrrrryyggrrrrrr"},
            {"duration": 6, "state": "rrrrrrGGrrrrrrGGrrrrrr"},
            {"duration": 4, "state": "rrrrrryyrrrrrryyrrrrrr"},
            {"duration": 1, "state": "rrrrrrrrrrrrrrrrrrrrrr"},
            {"duration": 31, "state": "gGggrrrrgGggrrrrGrrGrr"},
            {"duration": 5, "state": "gGggrrrrgGggrrrrrrrrrr"},
            {"duration": 4, "state": "yyyyrrrryyyyrrrrrrrrrr"},
            {"duration": 1, "state": "rrrrrrrrrrrrrrrrrrrrrr"}
        ]}

        # For a simplified action space, we can use 2 phase groups
        # Each group will consist of non-conflicting directions, with 4 seconds of yellow, then 1 second of red and 5 seconds of green at the end  
        # During the whole time that one group goes through its cycle, the other group will be in red.
        # Priority lefts are not treated with any priority in this setup.
        # The pedestrian crossing will be green during the green phase of the vehicle lanes in that group.
        # All right turns are always turned green. (This not implemented yet. Since the pedestrian frequency is relatively high)
        # If there is a switch, then the 4 seconds yellow and 1 second red will be applied to lanes (corresponding to other phase group). Hence thats the first thing that will occur in each phase group.
        self.phase_groups = {

            # Group 0
            # incoming vehicle lanes: north-straight, north-right, north-left, south-straight, south-right, south-left
            # pedestrian crossings: west, east
            0: [{"duration": 4, "state": "yyyyrrrryyyyrrrrrrrrrr"}, # These are to buffer the transition from the other one. Only necessary if there is a switch from another group.
                {"duration": 1, "state": "rrrrrrrrrrrrrrrrrrrrrr"},
                { "duration": 5, "state": "rrrrGGGGrrrrGGGGrrGrrG"}], # This is the actual green phase for this group

            # Group 1
            # incoming vehicle lanes: west-straight, west-right, east-straight, east-right, east-left
            # pedestrian crossings: south, north
            1: [{ "duration": 4, "state": "rrrryyyyrrrryyyyrrrrrr"}, # These are to buffer the transition from the other one. Only necessary if there is a switch from another group.
                { "duration": 1, "state": "rrrrrrrrrrrrrrrrrrrrrr"},
                { "duration": 5, "state": "GGGGrrrrGGGGrrrrGrrGrr"}] # This is the actual green phase for this group
            }

        self.current_phase_group = None
        self.current_state_index = None 
        self.corrected_occupancy_map = None

        self.tl_lane_dict = {}

        self.vehicle_input_trips = args.vehicle_input_trips
        self.vehicle_output_trips = args.vehicle_output_trips
        self.pedestrian_input_trips = args.pedestrian_input_trips
        self.pedestrian_output_trips = args.pedestrian_output_trips
        self.demand_scale_min = args.demand_scale_min
        self.demand_scale_max = args.demand_scale_max

        self.previous_action = None
        self.action_duration = args.action_duration  # Duration of each action in seconds
        # Number of simulation steps that should occur for each action. trying to ajuust for any given step length
        self.steps_per_action = int(self.action_duration / self.step_length) # This is also the size of the observation buffer
        self.observation_buffer_size = self.steps_per_action
        self.current_action_step = 0 # To track where we are within the curret action's duration

        self.directions = ['north', 'east', 'south', 'west']
        self.turns = ['straight', 'right', 'left']
        self.pressure_dict = {tl_id: {'vehicle': {}, 'pedestrian': {}} for tl_id in self.tl_ids}

    def _initialize_lanes(self,):
        """
        The outgoing lanes consist of 4 traffic outgoing lanes and three pedestrian lanes (after pedestrians have entered).
        Manual insertion of lanes based on natural classification.

        For Vehicles:
            Incoming lanes by direction
            Outgoing lanes by direction
        For Pedestrians:
            Incoming lanes by direction
            Outgoing lanes by direction

        Need to follow a protocol to represent the edge with edge. in front
        """

        # Manual insertion. If the same lane is used for multiple turn directions (straight, left) mark them with -1.
        self.tl_lane_dict['cluster_172228464_482708521_9687148201_9687148202_#5more'] = {
            "vehicle": {
                "incoming": {
                    # If there is a junction, then locate the edges in that junction and add the appropriate incoming edges here (exclude walking areas, or lanes already added).
                    # Because there is a zipper lane here # 'junction:172668791', using . as delimiter because : is in the name of the edge
                    "south-straight": ['1078803478#0_1', 'edge.:172668791_0','1078803477#5_1'] ,
                    "south-right": ['-1'],
                    "south-left": ['1078803478#0_2'],
                    "west-straight": ['1060131306#0_0'], # Same number with positive sign 2
                    "west-right": ['-1'] ,
                    "west-left": ['-1'],
                    "east-straight": ['-16666012#2_0'] , # A negative sign
                    "east-right": ['-1'], # The same lane in east-straight acts as right and left
                    "east-left": ['-1'],
                    "north-straight": ['773672648#0_1'], 
                    "north-right": ['773672648#0_0'],
                    "north-left": ['773672648#0_2'],
                },
                # Based on incoming from direction
                "inside": {
                    "south-straight": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_5'],
                    "south-right": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_4'],
                    "south-left": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_6'],
                    "west-straight": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_9'],
                    "west-right": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_8'],
                    "west-left": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_10'],
                    "east-straight": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_1'],
                    "east-right": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_0'],
                    "east-left": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_2'],
                    "north-straight": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_13'],
                    "north-right": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_12'],
                    "north-left": ['edge.:cluster_172228464_482708521_9687148201_9687148202_#5more_14'],
                },

                "outgoing": {
                    "west": ['-1060131306#0_0'] , # A negative sign 2
                    "south": ['773672649#1_0'],
                    "east": ['16666012#2_0'], # Same number with positive sign
                    "north": ['452522817#1_1'],
                },
            },
            "pedestrian": {
                "incoming": { # Incoming and want to cross this direction. From both sides
                    "west": [':cluster_172228464_482708521_9687148201_9687148202_#5more_w7',':cluster_172228464_482708521_9687148201_9687148202_#5more_w0'] ,
                    "south": [':cluster_172228464_482708521_9687148201_9687148202_#5more_w5', ':cluster_172228464_482708521_9687148201_9687148202_#5more_w6'],
                    "east": [':cluster_172228464_482708521_9687148201_9687148202_#5more_w4',':cluster_172228464_482708521_9687148201_9687148202_#5more_w3' ],
                    "north": [':cluster_172228464_482708521_9687148201_9687148202_#5more_w1',':cluster_172228464_482708521_9687148201_9687148202_#5more_w2'] ,
                },
                "outgoing": { # Is when they are inside the crossing
                    "west": [':cluster_172228464_482708521_9687148201_9687148202_#5more_c3'],
                    "south": [':cluster_172228464_482708521_9687148201_9687148202_#5more_c2'] ,
                    "east": [':cluster_172228464_482708521_9687148201_9687148202_#5more_c1'],
                    "north": [':cluster_172228464_482708521_9687148201_9687148202_#5more_c0'],
                },
            }
        }

    def _get_vehicle_direction(self, signal_state):
        """

        """
        # Define signal bits for left and right blinkers
        VEH_SIGNAL_BLINKER_RIGHT = 0b1  # Bit 0
        VEH_SIGNAL_BLINKER_LEFT = 0b10  # Bit 1

        # Check if left blinker is on
        left_blinker = bool(signal_state & VEH_SIGNAL_BLINKER_LEFT)
        
        # Check if right blinker is on
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

        Some things have to be done every step.
        1. Change the pedestrians's default color if they enter a crossing. They could be in the incoming walking area even when the exit the crossing.
            Get the occupancy map, if a person is in outgoing and their color is not already magenta, change the color to magenta.

        2. In case the same lanes are used for L, R, S turns (in case of vehicles and incoming). The straight lane will have repeated entries, remove them.  

        3. Vehicles are only included in the occupancy map if they are close to a given distance. In both incoming and outgoing directions.
        """
        
        # Handle outgoing pedestrians
        for tl_id in self.tl_ids:
            for _, persons in occupancy_map[tl_id]['pedestrian']['outgoing'].items():
                for person in persons:
                    if traci.person.getColor(person) != (255, 0, 255, 255):
                        traci.person.setColor(person, (255, 0, 255, 255))

        
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


        if print_map:
            print("\nOccupancy Map:")
            for tl_id, tl_data in occupancy_map.items():
                print(f"\nTraffic Light: {tl_id}")
                for agent_type in ["vehicle", "pedestrian"]:
                    print(f"  {agent_type.capitalize()}s:")
                    for direction in occupancy_map[tl_id][agent_type].keys():
                        print(f"    {direction.capitalize()}:")
                        for lane_group, ids in tl_data[agent_type][direction].items():
                            print(f"      {lane_group}: {len(ids)} [{', '.join(ids)}]")
                            for id in ids: 
                                if agent_type == "vehicle":
                                    distance = self._get_vehicle_distance_to_junction(tl_id, id)
                                    print(f"        {id}: {distance:.2f}m")
                                    

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
    
    def _update_outgoing_pressure_dict(self, corrected_occupancy_map):
        """
        Update the data str that holds info about pressure in outgoing directions.
        For both vehicles and pedestrians.
        """

        for tl_id in self.tl_ids:
            #### VEHICLES ####

            # Initialize pressure and calculate for each direction
            vehicle_pressure = {d: 0 for d in self.directions}

            for outgoing_direction in self.directions:
                # Calculate incoming traffic towards this direction
                incoming = 0
                if outgoing_direction == 'north':
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['south-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['east-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['west-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['south-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['east-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['west-left'])

                elif outgoing_direction == 'south':
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['north-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['east-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['west-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['north-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['east-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['west-right'])

                elif outgoing_direction == 'east':
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['west-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['north-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['incoming']['south-right'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['west-straight'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['north-left'])
                    incoming += len(corrected_occupancy_map[tl_id]['vehicle']['inside']['south-right'])

                elif outgoing_direction == 'west':
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

            #### PEDESTRIANS ####
            pedestrian_pressure = {d: 0 for d in self.directions}

            for outgoing_direction in self.directions:
                # Calculate incoming pedestrians towards this direction
                incoming = len(corrected_occupancy_map[tl_id]['pedestrian']['incoming'][outgoing_direction])
                
                # Calculate actual outgoing pedestrians
                outgoing = len(corrected_occupancy_map[tl_id]['pedestrian']['outgoing'][outgoing_direction])
                
                # Calculate pressure
                pedestrian_pressure[outgoing_direction] = incoming - outgoing
                self.pressure_dict[tl_id]['pedestrian'][outgoing_direction] = pedestrian_pressure[outgoing_direction]


    def _get_occupancy_map(self, ):
        """
        Features: 
            - If the same lane is used for multiple directions, the indicator light of vehicle is used to determine the direction. (The indicator light turns on about 100m far from the junction.)
        """

        occupancy_map = {}
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
                                        person_edge = traci.person.getRoadID(person)
                                        person_color = traci.person.getColor(person)
                                        if person_edge == lane:
                                            # print(f"Person: {person} Color: {traci.person.getColor(person)} " ) # By default this is the color.
                                            if direction == "incoming" and person_color == (255, 255, 0, 255):# And the color is still the default color (255, 255, 0, 255). 
                                                occupancy_map[tl_id][agent_type][direction][lane_group].append(person)

                                            else: # For outgoing, just being inside the crossing is enough.
                                                occupancy_map[tl_id][agent_type][direction][lane_group].append(person)

                                else: 
                                    print("Only implemented to work with JunctionDomain. Not implemented yet for external lanes or edges")

        return occupancy_map
    
    @property
    def action_space(self):
        """
        In the simplified action space with phase groups, the agents decision is binary.
        """
        num_actions = len(self.tl_ids) * len(self.phase_groups) # Number of traffic lights * Number of phase groups
        return gym.spaces.Discrete(num_actions)
        # return gym.spaces.Discrete(10) 

    @property
    def observation_space(self):
        """
        Each timestep (not action step) observation is the pressure in all outgoing directions.
        TODO: We do not have lane level granularity yet (in both action and observation).

        Observation space also needs to include the current traffic phase (or phase group in this case)
        """
        # 
        single_obs_shape = (len(self.tl_ids)*38, )

        # The observation is the entire observation buffer
        return gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(self.observation_buffer_size, *single_obs_shape),
            dtype=np.float32
        )
    
        # Previous dummy version
        # return gym.spaces.Box(low=0, high=1, shape=(1,))

    def step(self, action):
        """
        
        """

        if not self.sumo_running:
            raise Exception("Environment is not running. Call reset() to start the environment.")
        
        reward = 0
        done = False
        observation_buffer = []

        # Run simulation steps for the duration of the action
        for _ in range(self.steps_per_action):
            
            # Apply action needs to happen every timestep
            self._apply_action(action, self.current_action_step, self.previous_action)

            traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
            self.step_count += 1
            # Increment the current action step
            self.current_action_step = (self.current_action_step + 1) % self.steps_per_action # Wrapped around some modulo arithmetic

            # Collect observation at each substep
            obs = self._get_observation(print_map=False)
            #print(f"\nObservation: {obs}")
            observation_buffer.append(obs)

            self._update_outgoing_pressure_dict(self.corrected_occupancy_map)

            # Accumulate reward
            reward += self._get_reward(action)
            
            # Check if episode is done
            if self._check_done():
                done = True
                break

        formatted_buffer = "\n".join(f"{arr})" for arr in observation_buffer)
        print(f"\nAccumulated Observation:\n{formatted_buffer}")
        print(f"\nCurrent Action: {action}")
        print(f"\nAccumulated Reward: {reward}")

        self.previous_action = action
        # Show all edges in this junction: cluster_172228464_482708521_9687148201_9687148202_#5more
        # incoming_edges = traci.junction.getIncomingEdges('cluster_172228464_482708521_9687148201_9687148202_#5more')
        # outgoing_edges = traci.junction.getOutgoingEdges('cluster_172228464_482708521_9687148201_9687148202_#5more')

        # print(f"\nEdges: \tIncoming: {incoming_edges}\n\tOutgoing: {outgoing_edges}")

        # for tl in self.tl_ids:
        #     phase = traci.trafficlight.getPhase(tl)
        #     print(f"\n\tTraffic light {tl} is in phase {phase}")

        observation = np.asarray(observation_buffer)
        info = {}

        return observation, reward, done, False, info
        
    def _get_observation(self, print_map=False):
        """
        Generate an observation based on the current state of the environment.
        Previous Approach: Pressure(outgoing direction) = (Incoming traffic towards this direction including inside directions) - (Actual outgoing traffic in this direction)
        
        Not only phase group but also the subgroup.
        """
        
        # Get the occupancy map and print it
        occupancy_map = self._get_occupancy_map()
        self.corrected_occupancy_map = self._step_operations(occupancy_map, print_map=print_map, cutoff_distance=100)
        
        observation = []

        for tl_id in self.tl_ids:

            #### PHASE GROUP INFO ####
            observation.append(self.current_phase_group)
            observation.append(self.current_state_index/1) # To prevent a value of 2 [0, 1, 2] -> [0, 0.5, 1]

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

        # Create a mask for non-phase elements
        num_elements_per_tl = 2 + 12 + 12 + 4 + 4 + 4 # Total: 38
        mask = np.tile([False, False] + [True] * (num_elements_per_tl - 2), len(self.tl_ids))
        
        # Normalize
        normalizer = 10.0
        observation[mask] /= normalizer
        return observation
    
    def _apply_action(self, action, current_action_step, previous_action=None):
        """
        In the simplified action space with phase groups, previous action is used to determine if there was a switch.
        Duration is not set, a phase is set for the required duration.
        """
        # print(f"\nCurrent Action: {action}, Previous Action: {previous_action}")

        self.current_phase_group = action
        # For action space with phase groups
        if previous_action == None: # First action
            previous_action = action # Assume that there was no switch

        if action != previous_action:
            # print("Switching phase group")
            # Switch the phase group
            for tl_id in self.tl_ids:
                
                # All these shenanigans is to determine the index. TODO: Is there an efficient way to do this?
                durations = [phase["duration"] for phase in self.phase_groups[action]]
                cumulative_durations = [sum(durations[:i+1]) for i in range(len(durations))] # [4, 5, 10]
                for i, duration in enumerate(cumulative_durations):
                    if current_action_step < duration:
                        index = i
                        break
                self.current_state_index = index
                state = self.phase_groups[action][index]["state"]
                # print(f"Setting phase: {state}")
                traci.trafficlight.setRedYellowGreenState(tl_id, state)
                    
        else: # No switch. Just continue with the green in this phase group.
            # print("Continuing with the same phase group")
            self.current_state_index = 2

            for tl_id in self.tl_ids:
                    state = self.phase_groups[action][2]["state"] # Index is always 2
                    # print(f"Setting phase: {state}")
                    # Skip the first two phases, they are for buffering the transition.
                    traci.trafficlight.setRedYellowGreenState(tl_id, state)
                    
        
        # Previous version with 10 phases
        # for tl_id in self.tl_ids:
        #     traci.trafficlight.setPhase(tl_id, action)
        # self.current_phase = action

    def _get_reward(self, current_action):
        """
        - Consider both Vehicles and Pedestrians
        - Penalize frequent changes of action
        
        - Since pressure is calculated as incoming - outgoing. Pressure being high is not good.
        - To penalize high pressure, we make pressure negative.
        """

        reward = 0

        vehicle_pressure = 0
        pedestrian_pressure = 0

        for tl_id in self.tl_ids:
            
            # Use stored pressures for reward calculation
            for agent_type in ['vehicle', 'pedestrian']:
                for direction in self.directions:
                    pressure = self.pressure_dict[tl_id][agent_type][direction]

                    if agent_type == 'vehicle':
                        vehicle_pressure += pressure
                    else:
                        pedestrian_pressure += pressure

        reward = -0.5*vehicle_pressure - 0.5*pedestrian_pressure  

        # Frequency penalty
        if self.previous_action is not None and current_action != self.previous_action:
            reward -= 0.5  # Penalty for changing actions. Since this is per step reward. Action change is reflected multiplied by action steps.
            
        #print(f"\nStep Reward: {reward}")
        return reward

    def _check_done(self):
        """
        TODO: What more conditions can be added here?
        - Gridlock?
        - Crashes?
        """
        return self.step_count >= self.max_timesteps

    def reset(self, seed=None, options=None):
        """
        
        """

        super().reset(seed=seed)
        
        if self.sumo_running:
            traci.close()
        
        # Automatically scale demand (separately for pedestrian and vehicle)
        scale_factor_vehicle = random.uniform(self.demand_scale_min, self.demand_scale_max)
        scale_factor_pedestrian = random.uniform(self.demand_scale_min, self.demand_scale_max)

        scale_demand(self.vehicle_input_trips, self.vehicle_output_trips, scale_factor_vehicle, demand_type="vehicle")
        scale_demand(self.pedestrian_input_trips, self.pedestrian_output_trips, scale_factor_pedestrian, demand_type="pedestrian")

        if self.auto_start:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", 
                        "--start" , 
                        "--quit-on-end", 
                        "-c", "./craver.sumocfg", 
                        '--step-length', str(self.step_length)]
                        
        else:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", 
                        "--quit-on-end", 
                        "-c", "./craver.sumocfg", 
                        '--step-length', str(self.step_length)]
                        

        traci.start(sumo_cmd)
        self.sumo_running = True
        self.step_count = 0 # This counts the timesteps in an episode. Needs reset.

        self.current_action_step = 0
        self._initialize_lanes()

        # Randomly initialize current phase group 
        self.current_phase_group = random.choice(list(self.phase_groups.keys()))

        # Initialize the observation buffer
        observation_buffer = []
        for step in range(self.observation_buffer_size):

            # Apply the current phase group using _apply_action
            self._apply_action(self.current_phase_group, step, None)

            traci.simulationStep()
            obs = self._get_observation()
            observation_buffer.append(obs)

        observation = np.asarray(observation_buffer, dtype=np.float32)
        print(f"\nInitial observation inside: {observation}\n")
        print(f"\nInitial observation inside shape: {observation.shape}\n")
        info = {}
        return observation, info

    def close(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

