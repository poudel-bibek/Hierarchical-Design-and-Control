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

# TL_IDS, PHASES, PHASE_GROUPS, DIRECTIONS, TURNS, ORIGINAL_NET_FILE, CONTROLLED_CROSSWALKS_DICT, initialize_lanes

# Original 10 phases defaulted by SUMO with 93 seconds cycle time
# Keep this, required for TL only evaluation
PHASES = {
    'cluster_172228464_482708521_9687148201_9687148202_#5more': 
    [
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
    ]
}


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
CONTROLLED_CROSSWALKS_DICT = {
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
DIRECTIONS_AND_EDGES = {
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

# For a simplified action space, we can use 2 phase groups
# Each group will consist of non-conflicting directions, with 4 seconds of yellow, then 1 second of red and 5 seconds of green at the end  
# During the whole time that one group goes through its cycle, the other group will be in red.
# Priority lefts are not treated with any priority in this setup.
# The pedestrian crossing will be green during the green phase of the vehicle lanes in that group.
# All right turns are always turned green. (This not implemented yet. Since the pedestrian frequency is relatively high)
# If there is a switch, then the 4 seconds yellow and 1 second red will be applied to lanes (corresponding to other phase group). Hence thats the first thing that will occur in each phase group.
# The total sum of the 3 phases should be equal to action_duration. All Red and all Yellow times do not change when action_duration changes.
def get_phase_groups(action_duration):
    return {

        # Group 0
        # incoming vehicle lanes: north-straight, north-right, north-left, south-straight, south-right, south-left
        # pedestrian crossings: west, east
        0: [{"duration": 4, "state": "yyyyrrrryyyyrrrrrrrrrr"}, # These are to buffer the transition from the other one. Only necessary if there is a switch from another group.
            {"duration": 1, "state": "rrrrrrrrrrrrrrrrrrrrrr"},
            { "duration": (action_duration - (4 + 1)), "state": "rrrrGGGGrrrrGGGGrrGrrG"}], # This is the actual green phase for this group

        # Group 1
        # incoming vehicle lanes: west-straight, west-right, east-straight, east-right, east-left
        # pedestrian crossings: south, north
        1: [{ "duration": 4, "state": "rrrryyyyrrrryyyyrrrrrr"}, # These are to buffer the transition from the other one. Only necessary if there is a switch from another group.
            { "duration": 1, "state": "rrrrrrrrrrrrrrrrrrrrrr"},
            { "duration": (action_duration - (4 + 1)), "state": "GGGGrrrrGGGGrrrrGrrGrr"}] # This is the actual green phase for this group
    }

def initialize_lanes():
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
    return {
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

