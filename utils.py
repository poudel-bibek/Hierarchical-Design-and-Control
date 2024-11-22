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
    
    plt.title(f"Pedestrian Graph - Iteration {iteration}", fontsize=16)
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
        'edge': '#45B7D1',
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
    
    # Set axis limits with proper padding
    ax.set_xlim(x_min - x_range*padding, x_max + x_range*padding)
    ax.set_ylim(y_min - y_range*padding, y_max + y_range*padding)

    # Setup grid
    ax.grid(True, linestyle='--', color=colors['grid'], alpha=0.5)
    ax.set_axisbelow(True)

    # Draw edges with gradient effect
    edge_widths = [data['width'] for (_, _, data) in graph.edges(data=True)]
    max_width = max(edge_widths) if edge_widths else 1
    
    # Draw edges with gradient colors and glow effect
    for (u, v, data) in graph.edges(data=True):
        width = edge_width * (data['width']/max_width) if proportional_width else edge_width
        # Draw multiple lines with decreasing alpha for glow effect
        for w, a in zip([width*1.5, width*1.2, width], [0.1, 0.2, 0.7]):
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=[(u, v)],
                width=w,
                edge_color=colors['edge'],
                alpha=a,
                style='solid'
            )

    # Draw nodes with glow effect
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=colors['junction'],
        node_size=node_size*1.3,
        alpha=0.3,
        node_shape='o'
    )
    
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=colors['junction'],
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

    # Add legend elements
    legend_elements = [
        plt.Line2D([0], [0], color=colors['edge'], lw=edge_width, label='Path'),
        plt.scatter([0], [0], c=colors['junction'], marker='o', s=node_size, label='Junction')
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
