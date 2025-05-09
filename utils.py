import os
import xml
import time
import json
import torch
import logging
import random
import numpy as np
import seaborn as sns
from scipy import stats
import xml.etree.ElementTree as ET

def save_config(higher_ppo_args, lower_ppo_args, control_args, design_args, run_dir):
    """
    Save hyperparameters to json.
    """
    save_path = os.path.join(run_dir, 'config.json')
    config_to_save = {
        "hyperparameters": {
            "higher_ppo_args": higher_ppo_args,
            "lower_ppo_args": lower_ppo_args,
            "control_args": control_args,
            "design_args": design_args,
        },
    }
    with open(save_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)

def save_policy(higher_policy, lower_policy, lower_state_normalizer, norm_x, norm_y, save_path):  
    """
    """
    torch.save(
    {'higher': {
        'state_dict': higher_policy.state_dict(),  
        'norm_x': norm_x,
        'norm_y': norm_y
    },
    'lower': {
        'state_dict': lower_policy.state_dict(),  
        'state_normalizer_mean': lower_state_normalizer.mean.numpy(),  
        'state_normalizer_M2': lower_state_normalizer.M2.numpy(),  
        'state_normalizer_count': lower_state_normalizer.count.value  
    }}, save_path)

def load_policy(higher_policy, lower_policy, lower_state_normalizer, load_path):
    """
    Load policy state dict and welford normalizer stats.
    """
    checkpoint = torch.load(load_path)
    # In place operations
    higher_policy.load_state_dict(checkpoint['higher']['state_dict'])
    lower_policy.load_state_dict(checkpoint['lower']['state_dict'])
    lower_state_normalizer.manual_load(
        mean=torch.from_numpy(checkpoint['lower']['state_normalizer_mean']),  
        M2=torch.from_numpy(checkpoint['lower']['state_normalizer_M2']),  
        count=checkpoint['lower']['state_normalizer_count']
    )
    return checkpoint['higher']['norm_x'], checkpoint['higher']['norm_y']
    
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
    
    # print(f"{demand_type.capitalize()} demand scaled by factor {scale_factor}.") # Output written to {output_file}")
    
    # Wait for the file writing operations to finish (it could be large)
    time.sleep(1)

def clear_elements(parent, tag):
    """
    Helper to clear out old elements.
    """
    for elem in parent.findall(tag):
        parent.remove(elem)

def scale_demand_sliced_window(input_file, output_file, scale_factor, demand_type, window_size, evaluation=False):
    """
    Scale the demand in a randomly-sampled time window of the input file.
    window_size: max episode length (in seconds), including any warm‐up.
    A random t_start is chosen in [0, ORIGINAL_TIME_SPAN - window_size],
    and only trips/persons with depart in [t_start, t_start + window_size)
    are kept, re‐timed, and scaled.
    """
    
    if evaluation:
        START_SPAN = 2900
        END_SPAN = 3400 
    else: 
        START_SPAN = 0
        END_SPAN = 2400 

    t_start = random.uniform(START_SPAN, END_SPAN - window_size)
    t_end = t_start + window_size

    # 2) load XML and prepare root
    tree = ET.parse(input_file)
    root = tree.getroot()

    if demand_type == "vehicle":
        # remove all original trips, we'll re-add only the windowed ones
        clear_elements(root, "trip")

        # reload from input to filter
        full_tree = ET.parse(input_file)
        full_root = full_tree.getroot()
        all_trips = full_root.findall("trip")

        # first pass: collect trips inside window, shift & scale depart
        windowed = []
        for trip in all_trips:
            depart = float(trip.get("depart"))
            if t_start <= depart < t_end:
                # clone element
                new_trip = ET.Element("trip", trip.attrib)
                # shift so window start → 0, then scale down
                shifted = depart - t_start
                new_depart = shifted / scale_factor
                new_trip.set("depart", f"{new_depart:.2f}")
                windowed.append(new_trip)
                root.append(new_trip)

        # replicate across scale_factor slices
        original_count = len(windowed)
        for i in range(1, int(scale_factor)):
            for trip in windowed[:original_count]:
                dup = ET.Element("trip")
                for attr, val in trip.attrib.items():
                    if attr == "id":
                        dup.set(attr, f"{val}_{i}")
                    elif attr == "depart":
                        depart_val = float(val)
                        # spread duplicates over the window
                        offset = window_size * i / scale_factor
                        dup.set(attr, f"{depart_val + offset:.2f}")
                    else:
                        dup.set(attr, val)
                root.append(dup)

    elif demand_type == "pedestrian":
        # find the <routes> parent (or root if missing)
        routes_parent = root.find(".//routes") or root
        clear_elements(routes_parent, "person")

        # reload to filter
        full_tree = ET.parse(input_file)
        full_root = full_tree.getroot()
        all_persons = full_root.findall(".//person")

        windowed = []
        # first pass: filter, shift, scale
        for person in all_persons:
            depart = float(person.get("depart"))
            if t_start <= depart < t_end:
                new_person = ET.Element("person", person.attrib)
                shifted = depart - t_start
                new_depart = shifted / scale_factor
                new_person.set("depart", f"{new_depart:.2f}")

                # copy children, fixing 'from' if needed
                for child in person:
                    new_child = ET.SubElement(new_person, child.tag, child.attrib)
                    if child.tag == "walk" and "from" not in child.attrib:
                        edges = child.get("edges", "").split()
                        if edges:
                            new_child.set("from", edges[0])
                        else:
                            logging.warning(
                                f"Walk element for {new_person.get('id')} missing both 'from' and 'edges'."
                            )
                windowed.append(new_person)
                routes_parent.append(new_person)

        # replicate
        original_count = len(windowed)
        for i in range(1, int(scale_factor)):
            for person in windowed[:original_count]:
                dup = ET.Element("person")
                for attr, val in person.attrib.items():
                    if attr == "id":
                        dup.set(attr, f"{val}_{i}")
                    elif attr == "depart":
                        depart_val = float(val)
                        offset = window_size * i / scale_factor
                        dup.set(attr, f"{depart_val + offset:.2f}")
                    else:
                        dup.set(attr, val)

                # copy children
                for child in person:
                    new_child = ET.SubElement(dup, child.tag, child.attrib)
                    if child.tag == "walk" and "from" not in child.attrib:
                        edges = child.get("edges", "").split()
                        if edges:
                            new_child.set("from", edges[0])
                        else:
                            logging.warning(
                                f"Walk element for {dup.get('id')} missing both 'from' and 'edges'."
                            )
                routes_parent.append(dup)

    else:
        raise ValueError("Invalid demand_type: must be 'vehicle' or 'pedestrian'")

    # 3) serialize & write out
    xml_str = ET.tostring(root, encoding="unicode")
    dom = xml.dom.minidom.parseString(xml_str)
    pretty = "\n".join(line for line in dom.toprettyxml(indent="    ").split("\n") if line.strip())

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty)

    # ensure write has settled
    time.sleep(1)

def get_averages(result_json_path, total=False):
    """
    Helper function that reads a JSON file with results and returns the scales,
    means and standard deviations for :
    - Lower agent: Vehicle wait time, pedestrian wait time
    - Higher agent: Pedestrian arrival time
    """
    with open(result_json_path, 'r') as f:
        results = json.load(f)

    scales, veh_wait_mean, ped_wait_mean, ped_arrival_mean = [], [], [], []
    veh_wait_std, ped_wait_std, ped_arrival_std = [], [], []
    
    for scale_str, runs in results.items():
        scale = float(scale_str)
        scales.append(scale)
        veh_wait_vals = []
        ped_wait_vals = []
        ped_arrival_vals = []
        
        for run in runs.values():
            if total:
                veh_wait_vals.append(run["total_veh_waiting_time"])
                ped_wait_vals.append(run["total_ped_waiting_time"])
                ped_arrival_vals.append(run["total_ped_arrival_time"])
            else:
                veh_wait_vals.append(run["veh_avg_waiting_time"])
                ped_wait_vals.append(run["ped_avg_waiting_time"])
                ped_arrival_vals.append(run["average_arrival_time_per_ped"])
                
        veh_wait_mean.append(np.mean(veh_wait_vals))
        ped_wait_mean.append(np.mean(ped_wait_vals))
        ped_arrival_mean.append(np.mean(ped_arrival_vals))
        veh_wait_std.append(np.std(veh_wait_vals))
        ped_wait_std.append(np.std(ped_wait_vals))
        ped_arrival_std.append(np.std(ped_arrival_vals))

    # Convert to numpy arrays and sort by scale
    scales = np.array(scales)
    sort_idx = np.argsort(scales)
    
    return (scales[sort_idx], 
            np.array(veh_wait_mean)[sort_idx], 
            np.array(ped_wait_mean)[sort_idx],
            np.array(ped_arrival_mean)[sort_idx],
            np.array(veh_wait_std)[sort_idx],
            np.array(ped_wait_std)[sort_idx],
            np.array(ped_arrival_std)[sort_idx])

