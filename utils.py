import os
import time
import xml
import xml.etree.ElementTree as ET
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import random
import networkx as nx

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
                                proportional_width=False, 
                                show_edge_width=False,
                                show_coordinates=False,
                                scale_position='bottom_right'):
    """
    Creates an enhanced visualization of the pedestrian graph with vibrant colors
    
    Args:
        ...
        scale_position: Position of the scale bar ('bottom_right' or 'bottom_left')
    """
    # Set base font size
    BASE_FONT_SIZE = 20
    
    # Set the style
    sns.set_style("white")
    
    # Create a custom vibrant palette
    custom_palette = {
        'junction': '#FF6B6B',  # Coral red for junctions
        'crosswalk': '#4ECDC4',  # Turquoise for crosswalks
        'edge': '#45B7D1',  # Sky blue for edges
        'background': '#FFFFFF',  # Pure white background
        'text': '#2C3E50',  # Dark blue-grey for text
        'grid': '#E4E7EB'  # Light grey for grid
    }
    
    # Create figure with white background
    fig = plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.set_facecolor(custom_palette['background'])
    fig.patch.set_facecolor(custom_palette['background'])
    
    # Get position attributes
    pos = nx.get_node_attributes(graph, 'pos')
    
    # Calculate bounds for the plot
    x_coords = [coord[0] for coord in pos.values()]
    y_coords = [coord[1] for coord in pos.values()]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add some padding to the bounds
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    
    # Set the axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Create main axis with wider dashed grid
    ax.grid(True, 
            linestyle='--', 
            dashes=(8, 4),  # Longer dashes with wider gaps
            linewidth=0.8,  # Slightly thicker grid lines
            alpha=0.5,      # More visible grid
            color=custom_palette['grid'])
    
    # Make the grid appear behind other elements
    ax.set_axisbelow(True)
    
    # Identify crosswalk nodes
    crosswalk_nodes = [node for node in graph.nodes() if 'crosswalk' in str(node)]
    junction_nodes = [node for node in graph.nodes() if node not in crosswalk_nodes]
    
    # Draw edges with gradient effect
    edge_widths = [data['width'] for (u, v, data) in graph.edges(data=True)]
    max_width = max(edge_widths) if edge_widths else 1
    
    if proportional_width:
        normalized_widths = [3 * (w / max_width) for w in edge_widths]
    else:
        normalized_widths = [2.0] * len(edge_widths)  # Constant width for all edges
    
    # Create edge color gradient
    edge_colors = [sns.light_palette(custom_palette['edge'], n_colors=10)[int((w/max_width)*8)] 
                    for w in edge_widths] if proportional_width else \
                    [custom_palette['edge']] * len(edge_widths)
    
    # Draw edges with gradient colors and glow effect
    for (u, v, data), width, color in zip(graph.edges(data=True), normalized_widths, edge_colors):
        # Draw multiple lines with decreasing alpha for glow effect
        for w, a in zip([width*1.5, width*1.2, width], [0.1, 0.2, 0.7]):
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=[(u, v)],
                width=w,
                edge_color=color,
                alpha=a,
                style='solid',
                ax=ax
            )
    
    # Draw nodes with glow effect
    # First draw larger, more transparent nodes for glow
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=junction_nodes,
        node_color=custom_palette['junction'],
        node_size=400,
        alpha=0.3,
        node_shape='o',
        ax=ax
    )
    
    # Then draw the actual nodes
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=junction_nodes,
        node_color=custom_palette['junction'],
        node_size=300,
        alpha=0.9,
        node_shape='o',
        edgecolors='white',
        linewidths=2,
        ax=ax
    )
    
    if crosswalk_nodes:
        # Glow effect for crosswalk nodes
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=crosswalk_nodes,
            node_color=custom_palette['crosswalk'],
            node_size=500,
            alpha=0.3,
            node_shape='s',
            ax=ax
        )
        
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=crosswalk_nodes,
            node_color=custom_palette['crosswalk'],
            node_size=400,
            alpha=0.9,
            node_shape='s',
            edgecolors='white',
            linewidths=2,
            ax=ax
        )
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove x and y labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Add labels with better formatting
    if show_node_ids or show_coordinates:
        labels = {}
        for node, coords in pos.items():
            label_parts = []
            if show_node_ids:
                label_parts.append(str(node))
            if show_coordinates:
                label_parts.append(f"({coords[0]:.1f}, {coords[1]:.1f})")
            labels[node] = '\n'.join(label_parts)
            
        label_pos = {node: (coords[0], coords[1] + y_range * 0.02) for node, coords in pos.items()}
        nx.draw_networkx_labels(
            graph, label_pos,
            labels=labels,
            font_size=BASE_FONT_SIZE - 4,  # Node labels
            font_weight='bold',
            font_color=custom_palette['text'],
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
            ax=ax
        )
    
    # Add edge width annotations
    if show_edge_width and proportional_width:
        for (u, v, data) in graph.edges(data=True):
            edge_center = np.mean([pos[u], pos[v]], axis=0)
            plt.annotate(
                f"{data['width']:.1f}m",
                xy=edge_center,
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=BASE_FONT_SIZE - 4,  # Edge width labels
                color=custom_palette['text'],
                alpha=0.8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
            )
    
    # Combine network stats and legend in a single box with better organization
    combined_text = (
        f"Network Statistics:\n"
        f"  Nodes: {graph.number_of_nodes()}  •  Edges: {graph.number_of_edges()}\n"
        f"Display Options:\n"
        f"  Node IDs: {'✓' if show_node_ids else '✗'}  •  "
        f"Coordinates: {'✓' if show_coordinates else '✗'}  •  "
        f"Edge Widths: {'✓' if show_edge_width else '✗'}  •  "
        f"Proportional: {'✓' if proportional_width else '✗'}\n"
        f"Network Elements:\n"
        f"  \033[38;2;69;183;209m━━━\033[0m Pedestrian Path" + 
        (' (width proportional)' if proportional_width else '') + "  •  " +
        f"\033[38;2;255;107;107m●\033[0m Junction"
    )
    
    # Create a custom box for combined stats and legend
    stats_box = ax.text(
        0.5, 0.02,  # Position in axes coordinates (bottom center)
        combined_text,
        transform=ax.transAxes,
        fontsize=BASE_FONT_SIZE - 4,
        verticalalignment='bottom',
        horizontalalignment='center',
        bbox=dict(
            facecolor='white',
            edgecolor=custom_palette['junction'],
            alpha=0.9,
            pad=10,
            boxstyle='round,pad=1'
        ),
        family='monospace',
        color=custom_palette['text']
    )
    
    # Add colored symbols manually using plot for line and scatter for dot
    # These are just for the legend, positioned off-screen
    plt.plot([-1000, -999], [-1000, -999], color=custom_palette['edge'], 
            linewidth=3, label='Pedestrian Path')
    plt.scatter([-1000], [-1000], c=[custom_palette['junction']], 
                s=100, label='Junction')
    
    # Scale bar position and dimensions
    scale_bar_length = x_range / 10
    if scale_position == 'bottom_right':
        scale_bar_x = x_max - x_range * 0.15  # 15% from right edge
    else:  # bottom_left
        scale_bar_x = x_min + x_range * 0.05  # 5% from left edge
    scale_bar_y = y_min + y_range * 0.05  # 5% from bottom
    
    # Add "Scale" label
    plt.text(
        scale_bar_x + scale_bar_length/2,
        scale_bar_y + y_range * 0.02,  # Slightly above the scale bar
        'Scale',
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=BASE_FONT_SIZE - 4,
        color=custom_palette['text'],
        fontweight='bold'
    )
    
    # Draw scale bar with glow effect
    for w, a in zip([5, 3, 2], [0.1, 0.2, 1.0]):
        plt.plot(
            [scale_bar_x, scale_bar_x + scale_bar_length],
            [scale_bar_y, scale_bar_y],
            color=custom_palette['text'],
            linewidth=w,
            alpha=a,
            solid_capstyle='round'
        )
    
    # Scale measurement text
    plt.text(
        scale_bar_x + scale_bar_length/2,
        scale_bar_y - y_range * 0.02,
        f'{scale_bar_length:.1f}m',
        horizontalalignment='center',
        verticalalignment='top',
        fontsize=BASE_FONT_SIZE - 4,
        color=custom_palette['text'],
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    
    # Save with high quality
    plt.tight_layout()
    os.makedirs('graph_iterations', exist_ok=True)
    save_path = os.path.join('graph_iterations', f'enhanced_graph_iteration_{iteration}.png')
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        facecolor=fig.get_facecolor(),
        edgecolor='none'
    )
    print(f"Enhanced graph visualization saved to {save_path}")
    plt.close()
