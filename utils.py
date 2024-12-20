import os
import time
import xml
import xml.etree.ElementTree as ET
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import random
import networkx as nx
import numpy as np

def convert_demand_to_scale_factor(demand, demand_type, input_file):
    """
    Convert the demand to a scaling factor number.
    For vehicles: (veh/hr) that want to enter the network
    For pedestrians: (ped/hr) that want to enter the network
    """

    if demand <= 0:
        raise ValueError("Demand must be a positive number")
    
    if demand_type not in ['vehicle', 'pedestrian']:
        raise ValueError("Demand type must be either 'vehicle' or 'pedestrian'")
    
    # Calculate the original demand from the input file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    if demand_type == 'vehicle':
        original_demand = len(root.findall("trip"))
    else:  # pedestrian
        original_demand = len(root.findall(".//person"))
    
    if original_demand == 0:
        raise ValueError(f"No {demand_type} demand found in the input file")
    
    # Calculate the time span of the original demand
    if demand_type == 'vehicle':
        elements = root.findall("trip")
    else:
        elements = root.findall(".//person")
    
    # Find the start and end time of the demand
    start_time = min(float(elem.get('depart')) for elem in elements)
    end_time = max(float(elem.get('depart')) for elem in elements)
    time_span = (end_time - start_time) / 3600  # Convert to hours
    
    # Calculate the original demand per hour
    original_demand_per_hour = original_demand / time_span if time_span > 0 else 0
    print(f"\nOriginal {demand_type} demand per hour: {original_demand_per_hour:.2f}")

    if original_demand_per_hour == 0:
        raise ValueError(f"Cannot calculate original {demand_type} demand per hour")
    
    # Calculate the scale factor
    scale_factor = demand / original_demand_per_hour
    
    return scale_factor

def scale_demand(input_file, output_file, scale_factor, demand_type):
    """
    This function was causing some errors, so there is a new version as well.
    """
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    if demand_type == "vehicle":
        # Vehicle demand
        trips = root.findall("trip")
        for trip in trips:
            current_depart = float(trip.get('depart'))
            new_depart = current_depart / scale_factor
            trip.set('depart', f"{new_depart:.2f}")

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

    elif demand_type == "pedestrian":
        # Pedestrian demand
        persons = root.findall(".//person")
        for person in persons:
            current_depart = float(person.get('depart'))
            new_depart = current_depart / scale_factor
            person.set('depart', f"{new_depart:.2f}")

        original_person_count = len(persons)
        for i in range(1, int(scale_factor)):
            for person in persons[:original_person_count]:
                new_person = ET.Element('person')
                for attr, value in person.attrib.items():
                    if attr == 'id':
                        new_person.set(attr, f"{value}_{i}")
                    elif attr == 'depart':
                        new_depart = float(value) + (3600 * i / scale_factor)
                        new_person.set(attr, f"{new_depart:.2f}")
                    else:
                        new_person.set(attr, value)
                
                # Copy all child elements (like <walk>)
                for child in person:
                    new_child = ET.SubElement(new_person, child.tag, child.attrib)
                    # Ensure 'from' attribute is present for walk elements
                    if child.tag == 'walk' and 'from' not in child.attrib:
                        # If 'from' is missing, use the first edge in the route
                        edges = child.get('edges', '').split()
                        if edges:
                            new_child.set('from', edges[0])
                        else:
                            logging.warning(f"Walk element for person {new_person.get('id')} is missing both 'from' and 'edges' attributes.")
                
                # Find the correct parent to append the new person
                parent = root.find(".//routes")
                if parent is None:
                    parent = root
                parent.append(new_person)

    else:
        print("Invalid demand type. Please specify 'vehicle' or 'pedestrian'.")
        return

    # Convert to string
    xml_str = ET.tostring(root, encoding='unicode')
   
    # Pretty print the XML string
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="    ")
   
    # Remove extra newlines between elements
    pretty_xml_str = '\n'.join([line for line in pretty_xml_str.split('\n') if line.strip()])
    
    # If there are folders in the path that dont exist, create them
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the formatted XML to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)
    
    print(f"{demand_type.capitalize()} demand scaled by factor {scale_factor}.") # Output written to {output_file}")
    
    # Wait for the file writing operations to finish (it could be large)
    time.sleep(2)


def find_connecting_edges(net, start_edge_id, end_edge_id):
    """
    Use a breadth-first search to find paths between two edges.
    net = sumo network file.
    """
    start_edge = net.getEdge(start_edge_id)
    end_edge = net.getEdge(end_edge_id)
    
    
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

def create_new_sumocfg(network_iteration):
    """
    Need to iteratively load a new net file.
    """
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
                        <configuration>
                            <input>
                                <net-file value="network_iterations/network_iteration_{network_iteration}.net.xml"/>
                            </input>
                            <output>
                                <log value="sumo_logfile.txt"/>
                                <error-log value="sumo_errorlog.txt"/>
                            </output>
                        </configuration>"""
    
    temp_config_path = './SUMO_files/iterative_craver.sumocfg'
    with open(temp_config_path, 'w') as f:
        f.write(config_content)

def modify_net_file(crosswalks_to_disable, net_file_path):
    """
    Change the appearence of disallowed crosswalks.
    """
    tree = ET.parse(net_file_path)
    root = tree.getroot()

    for crosswalk_id in crosswalks_to_disable:
        # Find the edge element corresponding to this crosswalk
        edge = root.find(f".//edge[@id='{crosswalk_id}']")
        if edge is not None:
            # Find the lane within the crosswalk
            lane = edge.find('lane')
            if lane is not None:
                lane.set('width', '0.1')

    tree.write(net_file_path) # output


def save_graph_visualization(graph, iteration):
    """
    """

    plt.figure(figsize=(20, 15))

    pos = nx.get_node_attributes(graph, 'pos')
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=30, node_color='slateblue', alpha=0.8)
    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color='orange', width=2, alpha=0.2)
    # Draw node labels slightly above nodes
    label_pos = {node: (coords[0], coords[1] + random.uniform(-2.5, 2.5)) for node, coords in pos.items()}
    nx.draw_networkx_labels(graph, label_pos, font_size=6)
    
    plt.title(f"Pedestrian Graph - Iteration {iteration}", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs('graph_iterations', exist_ok=True)
    save_path = os.path.join('graph_iterations', f'graph_iteration_{iteration}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to {save_path}")
    plt.close()

def save_better_graph_visualization(graph, iteration, 
                                  show_node_ids=False, 
                                  show_coordinates=False,
                                  show_edge_width=False, 
                                  proportional_width=False,
                                  scale_position='bottom_right',
                                  node_size=300, 
                                  font_size=16,
                                  edge_width=2.0, 
                                  dpi=300):
    """
    Creates an enhanced visualization of the pedestrian graph.
    
    Args:
        graph: NetworkX graph to visualize
        iteration: Current iteration number for saving the file
        show_node_ids: If True, displays node IDs
        show_coordinates: If True, displays node (x,y) coordinates 
        show_edge_width: If True, displays edge width values in meters
        proportional_width: If True, draws edges with width proportional to actual width
        scale_position: Position of scale bar ('bottom_right' or 'bottom_left')
        node_size: Size of nodes in visualization
        font_size: Base font size for text
        edge_width: Base width for edges
        dpi: DPI for output image
    """
    # Set style and colors
    sns.set_style("white")
    colors = {
        'junction': '#FF6B6B',
        'special': '#4CAF50',  # Green color for special nodes
        'edge': '#45B7D1',
        'special_edge': '#90EE90',  # Light green for edges between special nodes
        'text': '#2C3E50',
        'grid': '#E4E7EB'
    }

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    pos = nx.get_node_attributes(graph, 'pos')
    
    # Calculate plot bounds with extra space at bottom
    x_coords, y_coords = zip(*pos.values())
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    ax.set_xlim(x_min - x_range*padding, x_max + x_range*padding)
    ax.set_ylim(y_min - y_range*padding, y_max + y_range*padding)

    ax.grid(True, linestyle='--', color=colors['grid'], alpha=0.5)
    ax.set_axisbelow(True)

    # Separate special and regular nodes
    special_nodes = [node for node in graph.nodes() if any(suffix in str(node) for suffix in ['_top', '_bottom', '_mid'])]
    regular_nodes = [node for node in graph.nodes() if node not in special_nodes]

    # Draw edges with different colors based on endpoint types
    edge_widths = [data['width'] for (_, _, data) in graph.edges(data=True)]
    max_width = max(edge_widths) if edge_widths else 1
    
    for (u, v, data) in graph.edges(data=True):
        width = edge_width * (data['width']/max_width) if proportional_width else edge_width
        # Check if both endpoints are special nodes
        is_special_edge = u in special_nodes and v in special_nodes
        edge_color = colors['special_edge'] if is_special_edge else colors['edge']
        
        # Draw multiple lines with decreasing alpha for glow effect
        for w, a in zip([width*1.5, width*1.2, width], [0.1, 0.2, 0.7]):
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=[(u, v)],
                width=w,
                edge_color=edge_color,
                alpha=a,
                style='solid'
            )

    # Draw regular nodes
    if regular_nodes:
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=regular_nodes,
            node_color=colors['junction'],
            node_size=node_size*1.3,
            alpha=0.3,
            node_shape='o'
        )
        
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=regular_nodes,
            node_color=colors['junction'],
            node_size=node_size,
            alpha=0.9,
            node_shape='o',
            edgecolors='white',
            linewidths=2
        )

    # Draw special nodes
    if special_nodes:
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=special_nodes,
            node_color=colors['special'],
            node_size=node_size*1.3,
            alpha=0.3,
            node_shape='o'
        )
        
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=special_nodes,
            node_color=colors['special'],
            node_size=node_size,
            alpha=0.9,
            node_shape='o',
            edgecolors='white',
            linewidths=2
        )

    # Add labels if requested
    if show_node_ids or show_coordinates:
        labels = {}
        for node, coords in pos.items():
            parts = []
            if show_node_ids:
                parts.append(str(node))
            if show_coordinates:
                parts.append(f"({coords[0]:.1f}, {coords[1]:.1f})")
            labels[node] = '\n'.join(parts)
            
        label_pos = {node: (coords[0], coords[1] + y_range*0.02) for node, coords in pos.items()}
        nx.draw_networkx_labels(graph, label_pos, labels=labels, font_size=font_size-4)

    # Add edge width annotations if requested
    if show_edge_width and proportional_width:
        for u, v, data in graph.edges(data=True):
            edge_center = np.mean([pos[u], pos[v]], axis=0)
            plt.annotate(f"{data['width']:.1f}m", xy=edge_center, xytext=(5, 5),
                        textcoords='offset points', fontsize=font_size-4)

    # Update legend elements to include special nodes
    legend_elements = [
        plt.Line2D([0], [0], color=colors['edge'], lw=edge_width, label = "Existing Edge"),
        plt.Line2D([0], [0], color=colors['special_edge'], lw=edge_width, label='New Edge'),
        plt.scatter([0], [0], c=colors['junction'], marker='o', s=node_size, label='Existing Node'),
        plt.scatter([0], [0], c=colors['special'], marker='o', s=node_size, label='New Node')
    ]
    
    # Add legend at the bottom
    ax.legend(handles=legend_elements, loc='lower left', fontsize=font_size)

    # Add network stats
    junction_count = len(graph.nodes())
    edge_count = graph.number_of_edges()
    stats = (f"Network Statistics\n"
            f"Junctions: {junction_count}\n"
            f"Total Paths: {edge_count}")
    ax.text(0.5, -0.1, stats, transform=ax.transAxes, fontsize=font_size,
            horizontalalignment='center', verticalalignment='top')

    # Add scale bar
    scale_bar_length = x_range/10
    scale_x = x_max - scale_bar_length - x_range*0.05
    scale_y = y_min + y_range*0.05
    
    ax.plot([scale_x, scale_x + scale_bar_length], [scale_y, scale_y], 
            color=colors['text'], linewidth=2)
    
    ax.text(scale_x + scale_bar_length/2, scale_y + y_range*0.02, 'Scale',
             ha='center', fontsize=font_size-2)
    ax.text(scale_x + scale_bar_length/2, scale_y - y_range*0.02, f'{scale_bar_length:.1f}m',
             ha='center', fontsize=font_size-2)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    # Save output
    os.makedirs('graph_iterations', exist_ok=True)
    save_path = os.path.join('graph_iterations', f'enhanced_graph_iteration_{iteration}.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Enhanced graph visualization saved to {save_path}")
    plt.close()


#### XML related Utils #####    
def get_initial_veh_edge_config(edge_root, node_root, node_coords):
    """
    """

    horizontal_edges_top_veh = ['-16666012#2', '-16666012#3', '-16666012#4', '-16666012#5', 
                                '-16666012#6', '-16666012#7', '-16666012#9', '-16666012#11', 
                                '-16666012#12', '-16666012#13', '-16666012#14', '-16666012#15', 
                                '-16666012#16', '-16666012#17']
    horizontal_edges_bottom_veh = ['16666012#2', '16666012#3', '16666012#4', '16666012#5',
                                    '16666012#6', '16666012#7', '16666012#9', '16666012#11',
                                    '16666012#12', '16666012#13', '16666012#14', '16666012#15',
                                    '16666012#16', '16666012#17']
    
    veh_edges = {
        'top': {},
        'bottom': {}
    }

    edges_dict = {edge.get('id'): edge for edge in edge_root.findall('edge')}
    for edge_id in horizontal_edges_top_veh:
        if edge_id in edges_dict:
            edge = edges_dict[edge_id]
            from_node = edge.get('from')
            to_node = edge.get('to')
            
            if from_node in node_coords and to_node in node_coords:
                lane = edge.find('lane')
                lane_shape = lane.get('shape') if lane is not None else None
                
                veh_edges['top'][edge_id] = {
                    'from': from_node,
                    'to': to_node,
                    'from_x': node_coords[from_node],
                    'to_x': node_coords[to_node],
                    'edge_shape': edge.get('shape'),
                    'lane_shape': lane_shape
                }
        else:
            print(f"\nEdge {edge_id} not found.")

    for edge_id in horizontal_edges_bottom_veh:
        edge = edge_root.find(f"edge[@id='{edge_id}']")
        if edge is not None:
            from_node = edge.get('from')
            to_node = edge.get('to')
            
            if from_node in node_coords and to_node in node_coords:
                lane = edge.find('lane')
                lane_shape = lane.get('shape') if lane is not None else None
                
                veh_edges['bottom'][edge_id] = {
                    'from': from_node,
                    'to': to_node,
                    'from_x': node_coords[from_node],
                    'to_x': node_coords[to_node],
                    'edge_shape': edge.get('shape'),
                    'lane_shape': lane_shape
                }
        else:
            print(f"\nEdge {edge_id} not found.")
    
    return veh_edges

def get_new_veh_edges_connections(middle_nodes_to_add, networkx_graph, original_edg_file, original_nod_file, conn_root):
    """
    Find which vehicle edges (both top and bottom) each middle node intersects with.
    Uses x-coordinate of middle node to find intersecting edges.
    """
    # Parse XML files
    edge_tree = ET.parse(original_edg_file)
    edge_root = edge_tree.getroot()
    node_tree = ET.parse(original_nod_file)
    node_root = node_tree.getroot()

    # Create a dictionary of node coordinates
    node_coords = {}
    for node in node_root.findall('node'):
        node_id = node.get('id')
        node_coords[node_id] = round(float(node.get('x')), 2)

    original_edges = get_initial_veh_edge_config(edge_root, node_root, node_coords)

    # This needs to aldo contain all the edges that could be connected to the initial edges.
    iterative_edges = {}
    for edge in edge_root.findall('edge'):
        edge_id = edge.get('id')
        attributes_dict = edge.attrib # only from and to will be used
        # Add from_x and to_x to the attributes_dict
        attributes_dict['from_x'] = node_coords[attributes_dict['from']]
        attributes_dict['to_x'] = node_coords[attributes_dict['to']]
        iterative_edges[edge_id] = attributes_dict
    #print(f"iterative_edges: {iterative_edges}\n")

    edges_to_remove = []
    edges_to_add = {'top': {}, 'bottom': {}}
    
    # As multiple middle nodes can intersect with the same vehicle edge. The splitting into edges has to happen iteratively.
    # In the same process, the connections (in the conn file) need to change as we go. 
    # The old edge in a connection could either be a to or a from edge. For the counterpart edge, find if its to the right or left of the old edge.
    # Find intersects for each middle node and then update the conn file.
    for i in range(len(middle_nodes_to_add)):
        m_node = middle_nodes_to_add[i]
        x_coord = round(networkx_graph.nodes[m_node]['pos'][0], 2)
        
        # Handle top edges and top connection
        for edge_id, edge_data in original_edges['top'].items():
            if edge_data['from_x'] >= x_coord >= edge_data['to_x']:  # greater than from and less than to

                connections_referenced_top = []
                for connection in conn_root.findall('connection'):
                    if connection.get('from') == edge_id or connection.get('to') == edge_id:
                        connections_referenced_top.append(connection)
                
                print(f"Top edge {edge_id} intersects with middle node {m_node} at x={x_coord:.2f} and is referenced in {connections_referenced_top} connections.\n")
                edges_to_remove.append(edge_id)
                
                # Split edge and lane shapes at intersection point
                edge_shape_points = [tuple(map(float, point.split(','))) 
                                    for point in edge_data['edge_shape'].split()]
                lane_shape_points = [tuple(map(float, point.split(','))) 
                                    for point in edge_data['lane_shape'].split()]
                
                # Find y-coordinates at intersection by linear interpolation
                edge_y = interpolate_y_coordinate(edge_shape_points, x_coord)
                lane_y = interpolate_y_coordinate(lane_shape_points, x_coord)
                
                # Split edge shape points (note: for top edges, right is from, left is to)
                right_edge_points = []
                left_edge_points = []
                for point in edge_shape_points:
                    if point[0] >= x_coord:
                        right_edge_points.append(point)
                    else:
                        left_edge_points.append(point)
                
                # Add intersection point to both sides
                right_edge_points.append((x_coord, edge_y))
                left_edge_points.insert(0, (x_coord, edge_y))
                
                # Split lane shape points
                right_lane_points = []
                left_lane_points = []
                for point in lane_shape_points:
                    if point[0] >= x_coord:
                        right_lane_points.append(point)
                    else:
                        left_lane_points.append(point)
                
                # Add intersection point to both sides
                right_lane_points.append((x_coord, lane_y))
                left_lane_points.insert(0, (x_coord, lane_y))
                
                # Convert points back to shape strings
                right_edge_shape = ' '.join(f"{x:.2f},{y:.2f}" for x, y in right_edge_points)
                left_edge_shape = ' '.join(f"{x:.2f},{y:.2f}" for x, y in left_edge_points)
                right_lane_shape = ' '.join(f"{x:.2f},{y:.2f}" for x, y in right_lane_points)
                left_lane_shape = ' '.join(f"{x:.2f},{y:.2f}" for x, y in left_lane_points)
                
                # Add new edges to edges_to_add
                # Right part of split (from original from to middle)
                right_edge_id_top = f"{edge_id}right{i}" # The same edge can be split multiple times. Value of i not neceaasrily corresponding to number of times split.
                edges_to_add['top'][right_edge_id_top] = {
                    'new_node': m_node,
                    'from': edge_data['from'],
                    'to': m_node,
                    'from_x': round(edge_data['from_x'], 2),
                    'to_x': x_coord,
                    'edge_shape': right_edge_shape,
                    'lane_shape': right_lane_shape
                }
                
                # Left part of split (from middle to original to)
                left_edge_id_top = f"{edge_id}left{i}" # The same edge can be split multiple times.
                edges_to_add['top'][left_edge_id_top] = {
                    'new_node': m_node,
                    'from': m_node,
                    'to': edge_data['to'],
                    'from_x': x_coord,
                    'to_x': round(edge_data['to_x'], 2),
                    'edge_shape': left_edge_shape,
                    'lane_shape': left_lane_shape
                }

                # Add the new edges to iterative_edges (because we need the latest edge data below)
                iterative_edges[right_edge_id_top] = edges_to_add['top'][right_edge_id_top]
                iterative_edges[left_edge_id_top] = edges_to_add['top'][left_edge_id_top]

                # Now add new connections to conn_root and remove old connections.
                for connection in connections_referenced_top:
                    from_edge, to_edge = connection.get('from'), connection.get('to') # Existing connection edge ids 
                    # A new connection is formed between the edge 1 = 'other than the current edge id' and egde 2 =one of either f"{edge_id}right" or f"{edge_id}left"
                    edge_1 = to_edge if edge_id == from_edge else from_edge
                    edge_1_n1_x, edge_1_n2_x = iterative_edges[edge_1]['from_x'], iterative_edges[edge_1]['to_x'] # x_coord of the nodes of edge_1
                    lowest_e1_x = min(edge_1_n1_x, edge_1_n2_x)

                    # To find edge_2 (whether left or right), we need to find where edge_1 is located (with respect to the current edge id)
                    # The logic is: if the smallest x_coordinate of nodes of edge_1 is greater than the smallest x_coordinate of the nodes ofcurrent edge id (the one that was split), then edge_2 is connected by the right side of the split.
                    # Draw this on a paper to understand.
                    current_edge_n1_x, current_edge_n2_x = iterative_edges[edge_id]['from_x'], iterative_edges[edge_id]['to_x']
                    if lowest_e1_x >= min(current_edge_n1_x, current_edge_n2_x): # greater than from and less than to (correct direction)
                        edge_2 = right_edge_id_top
                    else:
                        edge_2 = left_edge_id_top

                    # Add a new connection with attributes
                    attributes = {'from': edge_1, 'to': edge_2, 'fromLane': str(0), 'toLane': str(0)}
                    new_connection = ET.Element('connection', attributes)
                    new_connection.text = None  # Ensure there's no text content
                    new_connection.text = "\n\t\t"
                    conn_root.append(new_connection)


                # remove old connections
                for connection in connections_referenced_top:
                    conn_root.remove(connection)
                
                break
        
        # Check bottom edges and bottom connection
        for edge_id, edge_data in original_edges['bottom'].items():
            if edge_data['from_x'] <= x_coord <= edge_data['to_x']: # less than from and greater than to (reversed direction)
                connections_referenced_bottom = []
                for connection in conn_root.findall('connection'):
                    if connection.get('from') == edge_id or connection.get('to') == edge_id:
                        connections_referenced_bottom.append(connection)
                
                print(f"Bottom edge {edge_id} intersects with middle node {m_node} at x={x_coord:.2f} and is referenced in {connections_referenced_bottom} connections.\n")
                edges_to_remove.append(edge_id)
                
                # Split edge and lane shapes at intersection point
                edge_shape_points = [tuple(map(float, point.split(','))) 
                                    for point in edge_data['edge_shape'].split()]
                lane_shape_points = [tuple(map(float, point.split(','))) 
                                    for point in edge_data['lane_shape'].split()]
                
                # Find y-coordinates at intersection by linear interpolation
                edge_y = interpolate_y_coordinate(edge_shape_points, x_coord)
                lane_y = interpolate_y_coordinate(lane_shape_points, x_coord)
                
                # Split edge shape points
                left_edge_points = []
                right_edge_points = []
                for point in edge_shape_points:
                    if point[0] <= x_coord:
                        left_edge_points.append(point)
                    else:
                        right_edge_points.append(point)
                
                # Add intersection point to both sides
                left_edge_points.append((x_coord, edge_y))
                right_edge_points.insert(0, (x_coord, edge_y))
                
                # Split lane shape points
                left_lane_points = []
                right_lane_points = []
                for point in lane_shape_points:
                    if point[0] <= x_coord:
                        left_lane_points.append(point)
                    else:
                        right_lane_points.append(point)
                
                # Add intersection point to both sides
                left_lane_points.append((x_coord, lane_y))
                right_lane_points.insert(0, (x_coord, lane_y))
                
                # Convert points back to shape strings
                left_edge_shape = ' '.join(f"{x:.2f},{y:.2f}" for x, y in left_edge_points)
                right_edge_shape = ' '.join(f"{x:.2f},{y:.2f}" for x, y in right_edge_points)
                left_lane_shape = ' '.join(f"{x:.2f},{y:.2f}" for x, y in left_lane_points)
                right_lane_shape = ' '.join(f"{x:.2f},{y:.2f}" for x, y in right_lane_points)
                
                # Add new edges to edges_to_add
                # Left part of split
                left_edge_id_bottom = f"{edge_id}left{i}" # The same edge can be split multiple times.
                edges_to_add['bottom'][left_edge_id_bottom] = {
                    'new_node': m_node,
                    'from': edge_data['from'],
                    'to': m_node,
                    'from_x': round(edge_data['from_x'], 2),
                    'to_x': x_coord,
                    'edge_shape': left_edge_shape,
                    'lane_shape': left_lane_shape
                }
                
                # Right part of split
                right_edge_id_bottom = f"{edge_id}right{i}" # The same edge can be split multiple times.
                edges_to_add['bottom'][right_edge_id_bottom] = {
                    'new_node': m_node,
                    'from': m_node,
                    'to': edge_data['to'],
                    'from_x': x_coord,
                    'to_x': round(edge_data['to_x'], 2),
                    'edge_shape': right_edge_shape,
                    'lane_shape': right_lane_shape
                }

                # Add the new edges to iterative_edges (because we need the latest edge data below)
                iterative_edges[right_edge_id_bottom] = edges_to_add['bottom'][right_edge_id_bottom]
                iterative_edges[left_edge_id_bottom] = edges_to_add['bottom'][left_edge_id_bottom]

                # Now add new connections to conn_root and remove old connections.
                for connection in connections_referenced_bottom:
                    from_edge, to_edge = connection.get('from'), connection.get('to') # Existing connection edge ids 
                    # A new connection is formed between the edge 1 = 'other than the current edge id' and egde 2 =one of either f"{edge_id}right" or f"{edge_id}left"
                    edge_1 = to_edge if edge_id == from_edge else from_edge
                    edge_1_n1_x, edge_1_n2_x = iterative_edges[edge_1]['from_x'], iterative_edges[edge_1]['to_x'] # x_coord of the nodes of edge_1
                    lowest_e1_x = min(edge_1_n1_x, edge_1_n2_x)

                    # To find edge_2 (whether left or right), we need to find where edge_1 is located (with respect to the current edge id)
                    # The logic is: if the smallest x_coordinate of nodes of edge_1 is greater than the smallest x_coordinate of the nodes ofcurrent edge id (the one that was split), then edge_2 is connected by the right side of the split.
                    # Draw this on a paper to understand.
                    current_edge_n1_x, current_edge_n2_x = iterative_edges[edge_id]['from_x'], iterative_edges[edge_id]['to_x']
                    if lowest_e1_x <= min(current_edge_n1_x, current_edge_n2_x):  # reversed direction
                        edge_2 = right_edge_id_bottom
                    else:
                        edge_2 = left_edge_id_bottom

                    # Add a new connection with attributes
                    attributes = {'from': edge_1, 'to': edge_2, 'fromLane': str(0), 'toLane': str(0)}
                    new_connection = ET.Element('connection', attributes)
                    new_connection.text = None  # Ensure there's no text content
                    new_connection.tail  = "\n\t\t"
                    conn_root.append(new_connection)

                # remove old connections
                for connection in connections_referenced_bottom:
                    conn_root.remove(connection)
                
                break

    return edges_to_remove, edges_to_add, conn_root

def interpolate_y_coordinate(points, x):
    """
    Helper function to interpolate y-coordinate at given x between two points.
    """
    # Find the two points that bracket the x-coordinate
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 <= x <= x2:
            # Linear interpolation
            if x2 - x1 == 0:  # Avoid division by zero
                return y1
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    # If x is outside the range, return the y-coordinate of the closest point
    if x < points[0][0]:
        return points[0][1]
    return points[-1][1]