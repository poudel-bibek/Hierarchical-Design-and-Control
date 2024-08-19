import math
import time 
import traci
import torch
import random
import gymnasium as gym
import numpy as np
import xml.etree.ElementTree as ET
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

        TODO: Get network related stuff from a config file. To make the code more general purpose.
        """

        if args.manual_demand_veh is not None :
            # Convert the demand to scaling factor first
            scaling = convert_demand_to_scale_factor(args.manual_demand_veh, "vehicle", args.vehicle_input_trips)
            scale_demand(args.vehicle_input_trips, args.vehicle_output_trips, scaling, demand_type="vehicle")

        if args.manual_demand_ped is not None:
            # Convert the demand to scaling factor first
            scaling = convert_demand_to_scale_factor(args.manual_demand_ped, "pedestrian", args.pedestrian_input_trips)
            scale_demand(args.pedestrian_input_trips, args.pedestrian_output_trips, scaling, demand_type="pedestrian")

        self.use_gui = args.gui
        self.step_length = args.step_length
        self.max_timesteps = args.max_timesteps
        self.sumo_running = False
        self.step_count = 0
        self.auto_start = args.auto_start
        self.tl_ids = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # Only control this one for now
        
        # Original 10 phases defaulted by SUMO with 93 seconds cycle time
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
        self.tl_pedestrian_status = {} # For pedestrians related to crosswalks attached to TLS.

        self.vehicle_input_trips = args.vehicle_input_trips
        self.vehicle_output_trips = args.vehicle_output_trips
        self.pedestrian_input_trips = args.pedestrian_input_trips
        self.pedestrian_output_trips = args.pedestrian_output_trips
        self.demand_scale_min = args.demand_scale_min
        self.demand_scale_max = args.demand_scale_max

        self.previous_tl_action = None
        self.action_duration = args.action_duration  # Duration of each action in seconds
        # Number of simulation steps that should occur for each action. trying to ajuust for any given step length
        self.steps_per_action = int(self.action_duration / self.step_length) # This is also the size of the observation buffer
        self.observation_buffer_size = self.steps_per_action
        self.current_action_step = 0 # To track where we are within the curret action's duration

        self.directions = ['north', 'east', 'south', 'west']
        self.turns = ['straight', 'right', 'left']
        self.pressure_dict = {tl_id: {'vehicle': {}, 'pedestrian': {}} for tl_id in self.tl_ids}

        self.original_net_file = './SUMO_files/original_craver_road.net.xml'

        # dict. Ordered from left to right.
        # Should contain the ID as the crosswalk itself. A pedestrian's current_edge will end with _c0 if they are inside the crosswalk.  (Not present in the route)
        # If the pedestrian is already in the corsswalk, they dont need to be re-routed.

        # The vicinity walking edges from where pedestrians have to be re-routed. (Not present in the route)

        # The connected edges (Present in the route, are connected to the junctions) from which a couple step lookup ahead can be used to determine if the pedestrian is about to cross the crosswalk. 
        # Actually, the best place to do a one step lookup is from the walking area on top of the related junction itself. So connected_edges may not be used.
        # This cannot be done when the pedestrian is too far away. This is already where we are looking into something that cannot be determined observationally. 

        # Re-route edges (Present in the route) which acts as a focal point to ensure that pedestrians are infact crossing the crosswalk.
        # For controlled teleportation to the starting point.
        # Cannot put internal edges in reroute edges (wont be detected in teleportation). Also cannot put the edges after the junction, cannot find a route from there.

        # What is excluded is the walking areas in the junctions near the crosswalks. Pedestrians present here could be going anywhere.
        # _w0 or _w1 attached to junction names will be the walking areas over that junction. Pedestrians over these junctions will not be re-routed. 
        # However, pedestrians here might be important for observations later.
        self.controlled_crosswalks_dict = {
            0: { # This one is special case of a crosswalk (because of the double structure), the ones in the  middle should be excluded
                'ids': [':9687187500_c0', ':9687187501_c0'],
                'vicinity_walking_edges': [':9687187501_w1', '1054121747#2', ':9687187495_w0', '1058666192',':9687187500_w0', ':9687187501_w0'], # To re-route from
                'related_junction_edges': [':9727816658_w0', ':9687187526_w0'], # For forward lookup
                'connected_edges': ['1054121752#1', '1054121752#0', '1058666191#4', '1058666191#5'],
                'reroute_edges': {'upside': '1054121747#2' , 'downside': '1058666192' }, # Used as a fulcrum
            },
            # Crosswalk no. 1 and 2 are not controlled. They are simply present here to serve as a next best route if crosswalk 0 or 3 are disabled.
            1: {
                'ids': [':cluster_172228464_482708521_9687148201_9687148202_#5more_c2',],
                'vicinity_walking_edges': ['1054116929#4', '1054116929#2',':cluster_172228464_482708521_9687148201_9687148202_#5more_w6','1058666193_0',':cluster_172228464_482708521_9687148201_9687148202_#5more_w5' ], 
                'related_junction_edges': [':9687148199_w0','1054116933_0',':9687148198_w0',':9727816638_w0'],  
                'connected_edges': ['1078803478#0','1058666191#4','1058666191#3','1054116932#1','1054116932#0'],
                'reroute_edges': {'upside': '1054116929#4' , 'downside': '1054116929#2' }, 
            },
            2: {
                'ids': [':cluster_172228464_482708521_9687148201_9687148202_#5more_c1',],
                'vicinity_walking_edges': ['1054116929#0', '1054116929#1',':cluster_172228464_482708521_9687148201_9687148202_#5more_w4'], 
                'related_junction_edges': [':9687148197_w0',':9666242268_w0',':9727816638_w0',':9687148198_w0'],
                'connected_edges': ['1058666191#4','1058666191#3','1050677005#3','452522817#1','1050677005#2'],  
                'reroute_edges': {'upside': '1054116929#0' , 'downside': '1054116929#1' }, 
            },
            3: {
                'ids': [':9727816850_c0'],
                'vicinity_walking_edges': [':9727816850_w1', '1058666207#1', ':9727816844_w0',':9727816850_w0', '1058666206#0', '1058666207#0' ], # Have to be re-routed # JUnctions should not be here
                'related_junction_edges': [':9727816846_w0', ':9727816851_w0'], 
                'connected_edges': ['1050677005#7','1050677005#6','1058666191#1','1058666191#2'],
                'reroute_edges': {'upside': '1058666207#1' , 'downside': '1058666207#0' }, 
            },
            4: {
                'ids': [':9727816623_c0'],
                'vicinity_walking_edges': ['1058666188#1', '1051865729#3',':9727816623_w0', ':9727816623_w1'],
                'related_junction_edges': [':9727816625_w0', ':9666274798_w0'], # All edges with _w0 in the end begin with : in front
                'connected_edges': ['1058666187#2','1058666187#3', '1050677005#10','1050677005#9' ], 
                'reroute_edges': {'upside': '1051865729#3' , 'downside': '1058666188#1' },
            },
            5: {
                'ids': [':9740157155_c0'],
                'vicinity_walking_edges': ['1060131391#1', '1060131391#0', ':9740157155_w0',':9740157155_w1',':9740157153_w0', '1060131390' ],
                'related_junction_edges': [':9666274886_w0', ':9740157154_w0'],
                'connected_edges': ['1060131388#2','1060131388#3', '1050677005#13', '1050677005#12'],
                'reroute_edges': {'upside': '1060131391#1' , 'downside': '1060131391#0'}, 
            },
            6: {
                'ids': [':cluster_9740157181_9740483933_c0'],
                'vicinity_walking_edges': [':cluster_9740157181_9740483933_w0', ':cluster_9740157181_9740483933_w1', '1060131401#2', '1060131401#3'],
                'related_junction_edges': [':9740157180_w0', ':9655154530_w0'],
                'connected_edges': ['1060131402', ':9740483934_w0', ':9740157180_w0', '1060131401#1', '1050677005#14', '1050677005#13', '1050677007#1'],
                'reroute_edges': {'upside': '1060131401#3' , 'downside': '1060131401#2'},
            },
            7: {
                'ids': [':9740157194_c0'],
                'vicinity_walking_edges': ['1060131405#1', ':9740157194_w1', ':9740157194_w0', ':9740157192_w0', '1060131406', '1060131405#0'], # Have to be re-routed
                'related_junction_edges': [':9740157204_w0', ':9740157195_w0', ':10054309033_w0', ], # One step lookup
                'connected_edges': ['1050677005#16', '1098062395', '1050677005#18', '1060131403#1', '1060112727#1', '1060131404#1'],
                'reroute_edges': {'upside': '1060131406'  , 'downside': '1060131405#1'},
            },
            8: {
                'ids': [':9740157209_c0'],
                'vicinity_walking_edges': ['1060131408#1', ':9740157209_w0', ':9740157209_w1', '1060131408#0', '1060131410' ],
                'related_junction_edges': [':9740157207_w0', ':9740157211_w0', ':9740157210_w0', '1060131404#2' ], # For lookup
                'connected_edges': [':9740484420_w0', '1060131404#3', '1050677005#19', '1050677005#18', '1060131409#1' ],
                'reroute_edges': {'upside': '1060131408#1'  , 'downside': '1060131408#0' }, 
            },
            9: {
                'ids': [':9740484527_c0'],
                'vicinity_walking_edges': ['1060166260#1', ':9740484527_w0', ':9740484527_w1', '1050677005#21'],
                'related_junction_edges': [':9740484528_w0', ':9740484524_w0'],
                'connected_edges': ['1060166262#2', '1050677005#20', '1060112787#2', '1060112787#1'],
                'reroute_edges': {'upside': '1050677005#21' , 'downside': '1060166260#1' },
            },
            10: {
                'ids': [':cluster_172228408_9739966907_9739966910_c2'],
                'vicinity_walking_edges': [':cluster_172228408_9739966907_9739966910_w2', ':cluster_172228408_9739966907_9739966910_w3', '1060112789#1'], # Reroute. w3 covers both to the right and down from up
                'related_junction_edges': [':9739966908_w0', ':9739966904_w0', '1060112789#0', '1060112789#2'], # Lookup. 89#0 is the right one (downside)
                'connected_edges': [':9739966895_w0', ':9740484531_w0', '1060112790', ':cluster_172228408_9739966907_9739966910_w1'], # _w1 is the one on the right (downside)
                'reroute_edges': {'upside': '1060112789#2' , 'downside': '1060112789#1'},
            },
        }

        # Lookup the direction of the current edge.. based on current_edge (which contains internal edges.)
        # Since this can be used in the forward lookup as well. Including the internal edges in junctions.
        # Does not contain connected_edges
        self.direction_and_edges = {
            'upside': [ 
                '1054121747#2', ':9687187501_w1', ':9687187526_w0', ':9687187501_w0',
                '1054116929#4', ':9687148199_w0','1054116933_0',':cluster_172228464_482708521_9687148201_9687148202_#5more_w6',
                '1054116929#0', ':9687148197_w0',':9666242268_w0', '1054116930',
                ':9727816851_w0', '1058666207#1', ':9727816850_w1',
                ':9666274798_w0', '1051865729#3', ':9727816623_w0',
                ':9666274886_w0', '1060131391#1', ':9740157155_w0',
                ':cluster_9740157181_9740483933_w1', ':9655154530_w0', '1060131401#3',
                ':9740157194_w0', ':9740157195_w0', ':10054309033_w0', ':9740157192_w0', '1060131406', '1060131405#0',
                '1060131408#1', ':9740157209_w1', ':9740157210_w0',
                ':9740484527_w1', '1050677005#21', ':9740484524_w0',
                ':cluster_172228408_9739966907_9739966910_w3', '1060112789#2',
                ],

            'downside': [ 
                ':9727816658_w0', '1058666192', ':9687187495_w0', ':9687187500_w0',
                '1054116929#2', '1058666193_0',':9687148198_w0',':9727816638_w0',':cluster_172228464_482708521_9687148201_9687148202_#5more_w5',
                '1054116929#1', ':9727816638_w0',':9687148198_w0',':cluster_172228464_482708521_9687148201_9687148202_#5more_w4',
                ':9727816846_w0', ':9727816850_w0', ':9727816844_w0', '1058666207#0', '1058666206#0',
                ':9727816625_w0', '1058666188#1', ':9727816623_w1',
                ':9740157154_w0', '1060131391#0', ':9740157155_w1', ':9740157153_w0', '1060131390',
                '1060131402_0', ':9740157180_w0', ':cluster_9740157181_9740483933_w0', '1060131401#2',
                '1060131405#1', ':9740157194_w1', ':9740157204_w0',
                ':9740157209_w0', ':9740157207_w0', ':9740157211_w0', '1060131408#0', '1060131410','1060131404#2',
                ':9740484528_w0', '1060166260#1', ':9740484527_w0',
                ':9739966908_w0', ':9739966904_w0', '1060112789#0', ':cluster_172228408_9739966907_9739966910_w2', '1060112789#1',
                ],
            }
        
        self.current_crosswalk_selection = None 
        
        # Create a bunch of reverse lookup dictionaries which can be referenced in various ways
        self.crosswalk_to_disabled_vicinity_walking_edges = {
            crosswalk_id: data['vicinity_walking_edges']
            for _, data in self.controlled_crosswalks_dict.items()
            for crosswalk_id in data['ids']
        }

        # 
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

        # For crosswalk control 
        self.walking_edges_to_reroute_from = []
        self.related_junction_edges_to_lookup_from = []
        self.alternative_crosswalks_flat = []
        self.currently_rerouted = []
        self.alternative_crosswalks_num = []

        
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

    def find_connecting_edges(self, start_edge_id, end_edge_id):

        start_edge = self.net.getEdge(start_edge_id)
        end_edge = self.net.getEdge(end_edge_id)
        
        # Use a breadth-first search to find paths
        queue = [(start_edge, [start_edge])]
        visited = set()
        
        while queue:
            current_edge, path = queue.pop(0)
            
            if current_edge == end_edge:
                return path
            
            if current_edge in visited:
                continue
            
            visited.add(current_edge)
            
            for next_edge in current_edge.getOutgoing():
                if next_edge not in visited:
                    new_path = path + [next_edge]
                    queue.append((next_edge, new_path))
        
        return None  # No path found



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

            self._update_outgoing_pressure_dict(self.corrected_occupancy_map)

            # Accumulate reward
            reward += self._get_reward(current_tl_action)
            
            # Check if episode is done
            if self._check_done():
                done = True
                break

        formatted_buffer = "\n".join(f"{arr})" for arr in observation_buffer)
        print(f"\nAccumulated Observation:\n{formatted_buffer}")
        print(f"\nCurrent Action: {action}")
        print(f"\nAccumulated Reward: {reward}")

        self.previous_tl_action = current_tl_action
        observation = np.asarray(observation_buffer)
        info = {}

        return observation, reward, done, False, info
        
    def _get_observation(self, print_map=False):
        """
        Generate an observation based on the current state of the environment.
        Previous Approach: Pressure(outgoing direction) = (Incoming traffic towards this direction including inside directions) - (Actual outgoing traffic in this direction)
        
        Not only phase group but also the subgroup.
        Recently added: observation space also needs to include the previous action.
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
        
        # Normalize the items other than the masked ones
        normalizer = 10.0
        observation[mask] /= normalizer

        # Add previous action: 

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
            crosswalks_to_disable = []
            self.alternative_crosswalks_num = []
            
            current_crosswalk_action = action[1:].tolist() # The first element is the TL action
            # controlled_crosswalks_dict goes from 0 to 10, but we need to exclude 1 and 2
            for i in range(len(current_crosswalk_action)): 
                if current_crosswalk_action[i] == 0: # 0 means disable 

                    j = self.controlled_crosswalk_mask[i] # Get the actual crosswalk number (key), excludes 1 and 2
                    crosswalks_list = self.controlled_crosswalks_masked_dict[j]['ids'] # Get the crosswalks that are controlled by this action
                    for item in crosswalks_list: # Need a for loop because of them contains multiple crosswalks in a list
                        crosswalks_to_disable.append(item)

            ######################################
            # For testing purposes.
            # Based on the timesteps, allow a bunch and then disallow the bunch
            # print(f"Step count: {self.step_count}") # Increments by 10 here
            
            # # Define time ranges for disabling crosswalks
            # time_ranges = [
            #     (200, 1000, self.all_crosswalk_ids[3:7]),
            # ]

            # # Determine which crosswalks to disable based on current step
            # crosswalks_to_disable = [] 
            # for start, end, crosswalks in time_ranges:
            #     if start < self.step_count <= end:
            #         #print(f"\nTime range: {start} - {end}, Disabled Crosswalks: {crosswalks}")
            #         crosswalks_to_disable = crosswalks
            #         break
            
            # End of testing purposes code
            ######################################

            # self.all_crosswalks contains 1 and 2, controlled_crosswalks_masked_dict does not
            #print(f"All crosswalks: {self.controlled_crosswalk_masked_ids}\nCrosswalks to disable: {crosswalks_to_disable}")

            for crosswalk_id in crosswalks_to_disable:

                edges = self.crosswalk_to_disabled_vicinity_walking_edges[crosswalk_id] # Does not contain the crosswalk itself, just the vicinity walking edges list
                self.walking_edges_to_reroute_from.extend(edges)

                lookup_edges = self.crosswalk_to_related_junction_edges[crosswalk_id] # Just the lookup edges i.e., related_junction_edges
                self.related_junction_edges_to_lookup_from.extend(lookup_edges)
                
            #print(f"\nWalking edges to disable: {walking_edges_to_reroute_from}\n")
            #print(f"\nRelated junction edges to lookup from: {related_junction_edges_to_lookup_from}\n")

            # Find alternative crosswalks ids (they should include crosswalk 1 and 2, because 1 and 2 are present in the sim. we are controlling them as part of TL) and flatten
            alternative_crosswalks = [
                crosswalk_data['ids']
                for crosswalk_data in self.controlled_crosswalks_dict.values()
                if not any(cw in crosswalks_to_disable for cw in crosswalk_data['ids'])
            ]

            self.alternative_crosswalks_flat.extend([item for sublist in alternative_crosswalks for item in sublist])
            # Get the numerical crosswalk ids
            self.alternative_crosswalks_num = [self.edge_to_numerical_crosswalk_id[crosswalk_id] for crosswalk_id in self.alternative_crosswalks_flat]

            #print(f"\nAlternative crosswalks flattened: {self.alternative_crosswalks_flat}\n")

        # Although the crosswalks to disable are gotten every 10 timesteps, the enforcement of the action (i.e., re-routing pedestrians) is done every step.
        # This is going to be severely computationally taxing on the simulation.

        self._disallow_pedestrians(self.walking_edges_to_reroute_from,
                                                         self.related_junction_edges_to_lookup_from,
                                                        self.alternative_crosswalks_flat,)
        
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
        if self.previous_tl_action is not None and current_tl_action != self.previous_tl_action:
            reward -= 0.5  # Penalty for changing tl actions. Since this is per step reward. Action change is reflected multiplied by action steps.
            
        #print(f"\nStep Reward: {reward}")
        return reward

    def _check_done(self):
        """
        TODO: What more conditions can be added here?
        - Gridlock? of vehicles or pedestrians?
        - Crashes?
        """
        return self.step_count >= self.max_timesteps

    def _modify_net_file(self, crosswalks_to_disable):
        """
        Just for changing the appearence of disallowed crosswalks.
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

    def _disallow_pedestrians(self, walking_edges_to_reroute_from, related_junction_edges_to_lookup_from, alternate_crosswalks):
        """ 
        Disallow pedestrians means reroute pedestrians from the nearest possible crosswalk.
        This is called once per action i.e., after 10 actual simulation steps. 
        This means some pedestrians which have spawned and reached the crosswalks in the last 10 steps will not be rerouted. Which is unlikely.

        One important consideration is: 
        1. When to check if pedestrians need a re-route: whereever they may be at the action time step (because we go through all pedestrians) 
        2. When to actually perform the re-route: If they are present in the vicinity of the crosswalk. i.e., only if they are nearby, they will be able to see that a crosswalk is disabled. (closer to real-world scenario) 

        # We cannot check if a pedestrian far away has a crosswalk in their route (which they reach sometime in the future) and then re-route them immediately.
        # If the pedestrian in already in the crosswalk itself at the time of action, they dont need to be re-routed.
        """

        """
        Pedestrian current edge can be the crosswalk id itself.. or the walking areas associated. Both begin with :, Both will be invisible in the route but can be obtained by current edge.  
        Pedestrian can be in the internal edge (related to the junction) which will be _w0 or _w1 attached to the junction name. This can also be obtained using current edge.

        9.  Change pedestrian icon size
        
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
                    
                    print(f"\nPedestrian {ped_id} is being re-routed from crosswalk {current_crosswalk_num} to crosswalk {new_crosswalk_num} with ID: {new_crosswalk_id}")
                    print(f"Alternate crosswalk nums: {self.alternative_crosswalks_num}, differences: {differences}\n")

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
            traci.close()
        
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
                        ]
                        
        else:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", 
                        "--verbose",
                        "--quit-on-end", 
                        "-c", "./SUMO_files/craver.sumocfg", 
                        "--step-length", str(self.step_length),
                        ]
                        

        traci.start(sumo_cmd)

        # This should be done here after the SUMO call. As this can disallow pedestrians during the simulation run. 
        # Disallow pedestrians in some crosswalks. After sumo call beacuse we need traci.
        # disabled_crosswalks = self._disallow_pedestrians(to_disable)
        # print(f"\nDisabled crosswalks: {disabled_crosswalks}\n")

        self.sumo_running = True
        self.step_count = 0 # This counts the timesteps in an episode. Needs reset.

        self.current_action_step = 0
        self._initialize_lanes()

        # Randomly initialize the actions (current phase group and the current choice of crosswalks to activate) 
        self.current_phase_group = random.choice(list(self.phase_groups.keys()))
        self.current_crosswalk_selection = np.random.randint(2, size=len(self.controlled_crosswalks_masked_dict)).tolist() # 2 because of binary choice
        initial_action = torch.tensor([self.current_phase_group] + self.current_crosswalk_selection) # Make it a tensor so that its compatible with other outputs

        # Initialize the observation buffer
        observation_buffer = []
        for step in range(self.observation_buffer_size):

            # Apply the current phase group using _apply_action
            self._apply_action(initial_action, step, None)

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

