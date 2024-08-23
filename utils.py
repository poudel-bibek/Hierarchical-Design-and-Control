import xml.sax.saxutils as saxutils
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

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

def escape_attribute(value):
    return saxutils.quoteattr(str(value))

def is_valid_person(person):
    required_attrs = ['id', 'depart']
    return all(person.get(attr) is not None for attr in required_attrs)

def generate_unique_id(original_id, i):
    return f"{original_id}_scaled_{i}"

def scale_demand(input_file, output_file, scale_factor, demand_type):
    """
    Improved version of the scale_demand function.
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
        # Pedestrian demand (Improvements are here)
        persons = root.findall(".//person")
        valid_persons = [p for p in persons if is_valid_person(p)]
        
        for person in valid_persons:
            current_depart = float(person.get('depart'))
            new_depart = current_depart / scale_factor
            person.set('depart', f"{new_depart:.2f}")

        original_person_count = len(valid_persons)
        for i in range(1, int(scale_factor)):
            for person in valid_persons:
                new_person = ET.Element('person')
                for attr, value in person.attrib.items():
                    if attr == 'id':
                        new_id = generate_unique_id(value, i)
                        new_person.set(attr, new_id)
                    elif attr == 'depart':
                        new_depart = float(value) + (3600 * i / scale_factor)
                        new_person.set(attr, f"{new_depart:.2f}")
                    else:
                        new_person.set(attr, escape_attribute(value))
                
                for child in person:
                    new_child = ET.SubElement(new_person, child.tag)
                    for attr, value in child.attrib.items():
                        new_child.set(attr, escape_attribute(value))
                
                parent = root.find(".//routes")
                if parent is None:
                    parent = root
                parent.append(new_person)

def write_formatted_xml(root, output_file):
    def write_element(elem, file, indent=""):
        file.write(f"{indent}<{elem.tag}")
        for name, value in elem.attrib.items():
            file.write(f' {name}={escape_attribute(value)}')
        if len(elem) or elem.text:
            file.write(">\n")
            if elem.text:
                file.write(f"{indent}  {saxutils.escape(elem.text.strip())}\n")
            for child in elem:
                write_element(child, file, indent + "  ")
            file.write(f"{indent}</{elem.tag}>\n")
        else:
            file.write(" />\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        write_element(root, f)

def extract_pedestrian_crosswalks(sumo_network_xml: str, include_connections: bool = False) -> Tuple[List[Dict], Dict[int, Dict]]:
        tree = ET.parse(sumo_network_xml)
        root = tree.getroot()
        connections = []
        crosswalks = []

        # Look for edges with function="crossing"
        for edge in root.findall('.//edge[@function="crossing"]'):
            crosswalk = {
                'id': edge.get('id'),
                'function': 'crossing',
                'crossingEdges': edge.get('crossingEdges'),
            }
            
            # Get lane information
            lane = edge.find('lane')
            if lane is not None:
                crosswalk.update({
                    'allow': lane.get('allow'),
                    'speed': lane.get('speed'),
                    'length': lane.get('length'),
                    'width': lane.get('width'),
                    'shape': lane.get('shape')
                })
            
            crosswalks.append(crosswalk)

        # Look for connections with 'via' attribute containing 'c'
        if include_connections:
            for connection in root.findall('.//connection[@via]'):
                if 'c' in connection.get('via'):
                    conn = {
                        'id': connection.get('via'),
                        'from': connection.get('from'),
                        'to': connection.get('to'),
                        'fromLane': connection.get('fromLane'),
                        'toLane': connection.get('toLane')
                    }
                    connections.append(conn)
        
        return connections, crosswalks

# def scale_demand(input_file, output_file, scale_factor, demand_type):
#     """
#     This function was causing some errors, so there is a new version as well.
#     """
#     # Parse the XML file
#     tree = ET.parse(input_file)
#     root = tree.getroot()

#     if demand_type == "vehicle":
#         # Vehicle demand
#         trips = root.findall("trip")
#         for trip in trips:
#             current_depart = float(trip.get('depart'))
#             new_depart = current_depart / scale_factor
#             trip.set('depart', f"{new_depart:.2f}")

#         original_trip_count = len(trips)
#         for i in range(1, int(scale_factor)):
#             for trip in trips[:original_trip_count]:
#                 new_trip = ET.Element('trip')
#                 for attr, value in trip.attrib.items():
#                     if attr == 'id':
#                         new_trip.set(attr, f"{value}_{i}")
#                     elif attr == 'depart':
#                         new_depart = float(value) + (3600 * i / scale_factor)
#                         new_trip.set(attr, f"{new_depart:.2f}")
#                     else:
#                         new_trip.set(attr, value)
#                 root.append(new_trip)

#     elif demand_type == "pedestrian":
#         # Pedestrian demand
#         persons = root.findall(".//person")
#         for person in persons:
#             current_depart = float(person.get('depart'))
#             new_depart = current_depart / scale_factor
#             person.set('depart', f"{new_depart:.2f}")

#         original_person_count = len(persons)
#         for i in range(1, int(scale_factor)):
#             for person in persons[:original_person_count]:
#                 new_person = ET.Element('person')
#                 for attr, value in person.attrib.items():
#                     if attr == 'id':
#                         new_person.set(attr, f"{value}_{i}")
#                     elif attr == 'depart':
#                         new_depart = float(value) + (3600 * i / scale_factor)
#                         new_person.set(attr, f"{new_depart:.2f}")
#                     else:
#                         new_person.set(attr, value)
                
#                 # Copy all child elements (like <walk>)
#                 for child in person:
#                     new_child = ET.SubElement(new_person, child.tag, child.attrib)
                
#                 # Find the correct parent to append the new person
#                 parent = root.find(".//routes")
#                 if parent is None:
#                     parent = root
#                 parent.append(new_person)

#     else:
#         print("Invalid demand type. Please specify 'vehicle' or 'pedestrian'.")
#         return

#     # Convert to string
#     xml_str = ET.tostring(root, encoding='unicode')
   
#     # Pretty print the XML string
#     dom = xml.dom.minidom.parseString(xml_str)
#     pretty_xml_str = dom.toprettyxml(indent="    ")
   
#     # Remove extra newlines between elements
#     pretty_xml_str = '\n'.join([line for line in pretty_xml_str.split('\n') if line.strip()])
    
#     # Write the formatted XML to the output file
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write(pretty_xml_str)
    
#     print(f"{demand_type.capitalize()} demand scaled by factor {scale_factor}.") # Output written to {output_file}")
