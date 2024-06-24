import math
import traci
import gymnasium as gym
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import xml.dom.minidom

def scale_demand(input_file, output_file, scale_factor):
    """
    
    """
    
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find all trip elements
    trips = root.findall("trip")

    # Scale the depart times
    for trip in trips:
        current_depart = float(trip.get('depart'))
        new_depart = current_depart / scale_factor
        trip.set('depart', f"{new_depart:.2f}")

    # Create new trips
    original_trip_count = len(trips)
    for i in range(1, int(scale_factor)):
        for trip in trips[:original_trip_count]:
            new_trip = ET.Element('trip')
            for attr, value in trip.attrib.items():
                if attr == 'id':
                    new_trip.set(attr, f"{value}_{i}")
                elif attr == 'depart':
                    new_depart = float(value) + (3600 * i / scale_factor)
                    new_trip.set(attr, f"{new_depart:.2f}")
                else:
                    new_trip.set(attr, value)
            root.append(new_trip)

    # Convert to string
    xml_str = ET.tostring(root, encoding='unicode')
    
    # Pretty print the XML string
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="    ")
    
    # Remove extra newlines between trip elements
    pretty_xml_str = '\n'.join([line for line in pretty_xml_str.split('\n') if line.strip()])

    # Write the formatted XML to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)

    print(f"Demand scaled by factor {scale_factor}. Output written to {output_file}")

class CraverRoadEnv(gym.Env):
    def __init__(self, args):
        super().__init__()
        
        if args.scale_demand:
            scale_demand(args.input_trips, args.output_trips, args.scale_factor)

        self.use_gui = args.gui
        self.step_length = args.step_length
        self.max_steps = args.max_steps
        self.sumo_running = False
        self.step_count = 0
        self.auto_start = args.auto_start
        self.tl_ids = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # Only control this one for now
        
        self.phases = [
            {"duration": 32, "state": "rrrrgGggrrrrgGggrrGrrG"},
            {"duration": 5, "state": "rrrrgGggrrrrgGggrrrrrr"},
            {"duration": 4, "state": "rrrryyggrrrryyggrrrrrr"},
            {"duration": 6, "state": "rrrrrrGGrrrrrrGGrrrrrr"},
            {"duration": 4, "state": "rrrrrryyrrrrrryyrrrrrr"},
            {"duration": 1, "state": "rrrrrrrrrrrrrrrrrrrrrr"},
            {"duration": 31, "state": "gGggrrrrgGggrrrrGGrGGr"},
            {"duration": 5, "state": "gGggrrrrgGggrrrrrrrrrr"},
            {"duration": 4, "state": "yyyyrrrryyyyrrrrrrrrrr"},
            {"duration": 1, "state": "rrrrrrrrrrrrrrrrrrrrrr"}
        ]
        self.current_phase = 0
        self.tl_lane_dict = {}
        

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
                    for direction in ["incoming", "outgoing"]:
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
        
    def _get_occupancy_map(self, distance=100):
        """
        Features: 
            - If the same lane is used for multiple directions, the indicator light of vehicle is used to determine the direction. (The indicator light turns on about 100m far from the junction.)
        """

        occupancy_map = {}
        for tl_id, lanes in self.tl_lane_dict.items():
            occupancy_map[tl_id] = {
                "vehicle": {
                    "incoming": {},
                    "outgoing": {}
                },
                "pedestrian": {
                    "incoming": {},
                    "outgoing": {}
                }
            }
            
            for agent_type in ["vehicle", "pedestrian"]:
                for direction in ["incoming", "outgoing"]:
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
                                        print(f"Vehicle exists")
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

        """
        return gym.spaces.Discrete(10) 

    @property
    def observation_space(self):
        """

        """
        return gym.spaces.Box(low=0, high=1, shape=(1,))

    def step(self, action):
        if not self.sumo_running:
            raise Exception("Environment is not running. Call reset() to start the environment.")
        
        self._apply_action(action)
        traci.simulationStep()

        for tl in self.tl_ids:
            phase = traci.trafficlight.getPhase(tl)
            print(f"\n\tTraffic light {tl} is in phase {phase}")

        observation = self._get_observation(print_map=True)
        reward = self._get_reward()
        done = self._check_done()
        info = {}

        self.step_count += 1

        return observation, reward, done, False, info

    def _get_observation(self, print_map=False):

        # Get the occupancy map and print it
        occupancy_map = self._get_occupancy_map()
        corrected_occupancy_map = self._step_operations(occupancy_map, print_map=print_map, cutoff_distance=100)

        observation = np.zeros(self.observation_space.shape)
        return observation

    def _apply_action(self, action):
        for tl_id in self.tl_ids:
            traci.trafficlight.setPhase(tl_id, action)
        self.current_phase = action

    def _get_reward(self):
        reward = 0
        return reward

    def _check_done(self):
        return self.step_count >= self.max_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        

        if self.sumo_running:
            traci.close()
        
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
        self.step_count = 0
        self._initialize_lanes()

        observation = self._get_observation()
        info = {}
        return observation, info

    def close(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

# def main(args):

#     env = CraverRoadEnv(args)
#     observation, info = env.reset()

#     for _ in range(args.max_steps):
#         action = env.action_space.sample()
#         print(f"\nStep {env.step_count}: Taking action {action}")
#         observation, reward, terminated, truncated, info = env.step(action)

#         if terminated or truncated:
#             observation, info = env.reset()

#     env.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run SUMO traffic simulation with demand scaling.')
#     parser.add_argument('--gui', action='store_true', help='Use SUMO GUI (default: False)')
#     parser.add_argument('--step-length', type=float, default=1.0, help='Simulation step length (default: 1.0)')
#     parser.add_argument('--max_steps', type=int, default=2500, help='Maximum number of steps in one episode (default: 2000)')
#     parser.add_argument('--auto_start', action='store_true', help='Automatically start the simulation (default: False)')

#     parser.add_argument('--input_trips', type=str, default='./original_vehtrips.xml', help='Original Input trips file')
#     parser.add_argument('--output_trips', type=str, default='./scaled_vehtrips.xml', help='Output trips file')
#     parser.add_argument('--scale_demand', type=bool , default=True, help='Scale demand before starting the simulation')
#     parser.add_argument('--scale_factor', type=float, default=3.0, help='Demand scaling factor (default: 1.0)')
#     args = parser.parse_args()

#     main(args)