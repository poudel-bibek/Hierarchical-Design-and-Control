    
    update_count = 0
    action_timesteps = 0
    best_reward = float('-inf') 
    best_loss = float('inf')
    best_eval = float('inf')
    avg_eval = 200.0 # arbitrary large number
    eval_veh_avg_wait = 200.0 
    eval_ped_avg_wait = 200.0    

    # Every iteration, save all the sampled actions to a json file (by appending to the file).
    # A newer policy does importance sampling only every iteration. 
    actions_file_path = os.path.join(save_dir, f'sampled_actions.json')
    open(actions_file_path, 'w').close()
    sampled_actions = []

    all_memories = Memory()
    for iteration in range(0, total_iterations): # Starting from 1 to prevent policy update in the very first iteration.
        print(f"\nStarting iteration: {iteration + 1}/{total_iterations} with {global_step} total steps so far\n")
        
        old_policy = control_ppo.policy_old.to(device)
        old_policy.share_memory() # Dont pickle separate policy_old for each worker. Despite this, the old policy is still stale.
        old_policy.eval() # So that dropout, batnorm, laternote etc. are not used during inference

        #print(f"Shared policy weights: {control_ppo.policy_old.state_dict()}")
        train_queue = mp.Queue()
        train_processes = []
        active_train_workers = []
        for rank in range(control_args['num_processes']):

            worker_seed = SEED + iteration * 1000 + rank
            p = mp.Process(
                target=parallel_train_worker,
                args=(
                    rank,
                    old_policy,
                    control_args_worker,
                    train_queue,
                    worker_seed,
                    shared_state_normalizer,
                    shared_reward_normalizer,
                    device)
                )
            p.start()
            train_processes.append(p)
            active_train_workers.append(rank)
        
        while active_train_workers:
            print(f"Active workers: {active_train_workers}")
            rank, memory = train_queue.get()

            if memory is None:
                print(f"Worker {rank} finished")
                active_train_workers.remove(rank)
            else:
                current_action_timesteps = len(memory.states)
                print(f"Memory from worker {rank} received. Memory size: {current_action_timesteps}")
                all_memories.actions.extend(torch.from_numpy(np.asarray(memory.actions)))
                all_memories.states.extend(torch.from_numpy(np.asarray(memory.states)))
                all_memories.values.extend(memory.values)
                all_memories.logprobs.extend(memory.logprobs)
                all_memories.rewards.extend(memory.rewards)
                all_memories.is_terminals.extend(memory.is_terminals)

                sampled_actions.append(memory.actions[0].tolist())
                action_timesteps += current_action_timesteps
                global_step += current_action_timesteps * train_config['action_duration'] 
                print(f"Action timesteps: {action_timesteps}, global step: {global_step}")
                del memory #https://pytorch.org/docs/stable/multiprocessing.html

                # Update PPO every n times (or close to n) action has been taken 
                if action_timesteps >= control_args['update_freq']:
                    print(f"Updating PPO with {len(all_memories.actions)} memories") 

                    update_count += 1
                    # Anneal after every update
                    if control_args['anneal_lr']:
                        current_lr = control_ppo.update_learning_rate(update_count, total_updates)

                    avg_reward = sum(all_memories.rewards) / len(all_memories.rewards)
                    print(f"\nAverage Reward (across all memories): {avg_reward}\n")
                    #print(f"\nAll memories rewards: {all_memories.rewards}")

                    loss = control_ppo.update(all_memories)

                    # Reset all memories
                    del all_memories
                    all_memories = Memory() 
                    action_timesteps = 0
                    print(f"Size of all memories after update: {len(all_memories.actions)}")

                    # Save both during sweep and non-sweep
                    # Save (and evaluate the latest policy) every save_freq updates
                    if update_count % control_args['save_freq'] == 0:
                        latest_policy_path = os.path.join(control_args['save_dir'], f'policy_at_step_{global_step}.pth')
                        save_policy(control_ppo.policy, shared_state_normalizer, latest_policy_path)
                    
                        print(f"Evaluating policy: {latest_policy_path} at step {global_step}")
                        eval_json = eval(control_args_worker, ppo_args, eval_args, policy_path=latest_policy_path, tl= False) # which policy to evaluate?
                        _, eval_veh_avg_wait, eval_ped_avg_wait, _, _ = get_averages(eval_json)
                        eval_veh_avg_wait = np.mean(eval_veh_avg_wait)
                        eval_ped_avg_wait = np.mean(eval_ped_avg_wait)
                        avg_eval = ((eval_veh_avg_wait + eval_ped_avg_wait) / 2)
                        print(f"Eval veh avg wait: {eval_veh_avg_wait}, eval ped avg wait: {eval_ped_avg_wait}, avg eval: {avg_eval}")

                    # Save best policies 
                    if avg_reward > best_reward:
                        save_policy(control_ppo.policy, shared_state_normalizer, os.path.join(control_args['save_dir'], 'best_reward_policy.pth'))
                        best_reward = avg_reward
                    if loss['total_loss'] < best_loss:
                        save_policy(control_ppo.policy, shared_state_normalizer, os.path.join(control_args['save_dir'], 'best_loss_policy.pth'))
                        best_loss = loss['total_loss']
                    if avg_eval < best_eval:
                        save_policy(control_ppo.policy, shared_state_normalizer, os.path.join(control_args['save_dir'], 'best_eval_policy.pth'))
                        best_eval = avg_eval

                    # logging
                    if is_sweep: # Wandb for hyperparameter tuning
                        wandb.log({ "iteration": iteration,
                                        "avg_reward": avg_reward, # Set as maximize in the sweep config
                                        "update_count": update_count,
                                        "policy_loss": loss['policy_loss'],
                                        "value_loss": loss['value_loss'], 
                                        "entropy_loss": loss['entropy_loss'],
                                        "total_loss": loss['total_loss'],
                                        "current_lr": current_lr if control_args['anneal_lr'] else ppo_args['lr'],
                                        "approx_kl": loss['approx_kl'],
                                        "eval_veh_avg_wait": eval_veh_avg_wait,
                                        "eval_ped_avg_wait": eval_ped_avg_wait,
                                        "avg_eval": avg_eval,
                                        "global_step": global_step })
                        
                    else: # Tensorboard for regular training
                        writer.add_scalar('Training/Average_Reward', avg_reward, global_step)
                        writer.add_scalar('Training/Total_Policy_Updates', update_count, global_step)
                        writer.add_scalar('Training/Policy_Loss', loss['policy_loss'], global_step)
                        writer.add_scalar('Training/Value_Loss', loss['value_loss'], global_step)
                        writer.add_scalar('Training/Entropy_Loss', loss['entropy_loss'], global_step)
                        writer.add_scalar('Training/Total_Loss', loss['total_loss'], global_step)
                        writer.add_scalar('Training/Current_LR', current_lr if control_args['anneal_lr'] else ppo_args['lr'], global_step)
                        writer.add_scalar('Training/Approx_KL', loss['approx_kl'], global_step)
                        writer.add_scalar('Evaluation/Veh_Avg_Wait', eval_veh_avg_wait, global_step)
                        writer.add_scalar('Evaluation/Ped_Avg_Wait', eval_ped_avg_wait, global_step)
                        writer.add_scalar('Evaluation/Avg_Eval', avg_eval, global_step)
                    print(f"\nLogged data at step {global_step}\n")

                    # At the end of update, save normalizer stats
                    state_normalizer_mean = shared_state_normalizer.mean.numpy()  
                    state_normalizer_M2 = shared_state_normalizer.M2.numpy()  
                    state_normalizer_count = shared_state_normalizer.count.value  

        # Clean up. The join() method ensures that the main program waits for all processes to complete before continuing.
        for p in train_processes:
            p.join() 
        print(f"All processes joined\n\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del train_queue

        # Save all the sampled actions to a json file
        with open(actions_file_path, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            data[iteration] = sampled_actions
            f.seek(0)
            #print(f"Sampled actions: {sampled_actions}")
            json.dump(data, f, indent=4)
            f.truncate()
            f.close()
        sampled_actions = []

    if not is_sweep:
        writer.close()


















    def _update_xml_files(self, networkx_graph, iteration):
        """
        Update the XML component files to reflect the current state of the networkx graph.
        For base, use the "original" XML component files. For other iterations, use the "base" XML component files as a foundation and add/ remove elements.
        Iterative component files are saved in components directory.
        Iterative net files are saved in network_iterations directory.

        Networkx graph will already have:
        - End nodes with position values that come from the proposal.
        - Middle nodes to create traffic lights. Every proposal will have nodes with id _mid.
                - The _mid nodes need to be connected to vehicle edges on either side as well.
                - For every _mid node, a TL logic needs to be added to the traffic_light XML file.

        For iteration = base, the end nodes are already present in the XML component files.
        For other iterations, the end nodes will require reconstruction.

        For the nodes and edges related to pedestrian network:
            Remove them from the XML component files: If they don't exist in networkx graph.
            Add them to the XML component files: If they exist in networkx graph but don't exist in component XML.

        Node attributes:
            - <node id=" " x=" " y=" " type=" " />
            - For the middle nodes, type will "traffic_light" and an attribute tl =" " with the value same as id.
            - For the end nodes, type will be "dead_end"
        Edge attributes:
            - From the middle node to end nodes of type "highway.footway"
            - From middle node to vehicle nodes of type "highway.tertiary"
            - Both of these are needed because the traffic light is coordinating vehicles and pedestrians in the crosswalk.
            - <edge id=" " from=" " to=" " priority="1" type="highway.footway" numLanes="1" speed="2.78" shape=" " spreadType="center" width="2.0" allow="pedestrian">
            - shape seems difficult to get right.
            - create a nested lane element: <lane index="0" allow="pedestrian" width=" " speed=" ">
            - create a nested param element: <param key="origId" value=" "/>
            - end with </lane></edge>
        """

        # Parse the XML files
        prefix = "original" if iteration == 'base' else "iteration_base" # Every iteration will have the same base XML files.

        node_file = f'{self.component_dir}/{prefix}.nod.xml'
        node_tree = ET.parse(node_file)
        node_root = node_tree.getroot()

        edge_file = f'{self.component_dir}/{prefix}.edg.xml'
        edge_tree = ET.parse(edge_file)
        edge_root = edge_tree.getroot()

        connection_file = f'{self.component_dir}/{prefix}.con.xml'
        connection_tree = ET.parse(connection_file)
        connection_root = connection_tree.getroot() # This will be modified by get_new_veh_edges_connections

        traffic_light_file = f'{self.component_dir}/{prefix}.tll.xml'
        traffic_light_tree = ET.parse(traffic_light_file)
        traffic_light_root = traffic_light_tree.getroot()

        type_file = f'{self.component_dir}/{prefix}.typ.xml'
        type_tree = ET.parse(type_file)

        # Find ALL the nodes and edges in the XML component files (nod.xml and edg.xml)
        nodes_in_xml = { n.get('id'): n for n in node_root.findall('node') } # save the node element itself.
        edges_in_xml = { (e.get('from'), e.get('to')): e for e in edge_root.findall('edge') } # save the from, to nodes and edge element.

        # Find PEDESTRIAN nodes and edges in the XML component .edg file.
        pedestrian_edges_in_xml = {}
        pedestrian_nodes_in_xml = set()
        for (f, t), e in edges_in_xml.items():
            e_type = e.get('type')
            allow = e.get('allow', '')
            if e_type in ['highway.footway', 'highway.steps'] and 'pedestrian' in allow:
                pedestrian_edges_in_xml[(f, t)] = e
                pedestrian_nodes_in_xml.update([f, t]) # From node id alone we cant differentiate between vehicle and pedestrian nodes.

        # Extract pedestrian nodes and edges from networkx_graph
        pedestrian_nodes_in_graph = set(networkx_graph.nodes())
        # print(f"Pedestrian nodes in XML: {pedestrian_nodes_in_xml}\n")
        # print(f"Pedestrian nodes in graph: {pedestrian_nodes_in_graph}\n")
        # print(f"Pedestrian edges in XML: {list(pedestrian_edges_in_xml.keys())}\n")
        # print(f"Pedestrian edges in graph: {set(networkx_graph.edges())}\n")

        # Remove PEDESTRIAN nodes that are in XML component file but not in networkx graph.
        potential_nodes_to_remove = pedestrian_nodes_in_xml - pedestrian_nodes_in_graph
        # print(f"Potential Nodes to remove: Total: {len(potential_nodes_to_remove)},\n {potential_nodes_to_remove}\n")

        # Some edges may still access the nodes that are in potential_nodes_to_remove.
        # Find the edges that still access the nodes that are in potential_nodes_to_remove.
        edges_in_xml_that_access_removal_nodes = {}
        for (f, t) in edges_in_xml:
            if f in potential_nodes_to_remove or t in potential_nodes_to_remove:
                edges_in_xml_that_access_removal_nodes[(f, t)] = edges_in_xml[(f, t)] # These can be vehicle edges as well.

        # print(f"Edges in XML that still access the potential removal nodes: Total: {len(edges_in_xml_that_access_removal_nodes)}")
        # for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
        #     print(f"Edge: {f} -> {t}")
        #     print(f"Edge attributes: {e.attrib}\n")

        # In the edges that access nodes in potential_nodes_to_remove, some of the edges are vehicle edges (For e.g., when the old TL was removed).
        vehicle_edges_that_access_removal_nodes = {}
        for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
            e_type = e.get('type')
            disallow = e.get('disallow', '')
            if e_type == 'highway.tertiary' and 'pedestrian' in disallow: # vehicle edge attributes: highway.tertiary and disallowed pedestrian
                vehicle_edges_that_access_removal_nodes[(f, t)] = e
        # print(f"Vehicle edges that access removal nodes: Total: {len(vehicle_edges_that_access_removal_nodes)},\n {vehicle_edges_that_access_removal_nodes}\n")

        # Get all nodes that appear in vehicle edges
        nodes_in_vehicle_edges = set()
        for f, t in vehicle_edges_that_access_removal_nodes.keys():
            nodes_in_vehicle_edges.add(f)
            nodes_in_vehicle_edges.add(t)
        # print(f"Potential nodes to be removed: {potential_nodes_to_remove}\n Nodes in vehicle edges: {nodes_in_vehicle_edges}\n")

        # The nodes that appear in vehicle edges can be removed because they are not needed for the pedestrian network. Remove them
        pedestrian_nodes_to_remove = potential_nodes_to_remove - nodes_in_vehicle_edges
        # print(f"Actual pedestrian nodes to remove: Total: {len(pedestrian_nodes_to_remove)},\n {pedestrian_nodes_to_remove}\n")

        # Some pedestrian edges (at crossings) link to vehicle edges. Remove the pedestrian edges that are not linked to the vehicle edges.
        pedestrian_edges_to_remove = {}
        for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
            if (f, t) not in vehicle_edges_that_access_removal_nodes:
                pedestrian_edges_to_remove[(f, t)] = e
        # print(f"Actual pedestrian edges to remove: Total: {len(pedestrian_edges_to_remove)}, \n {pedestrian_edges_to_remove}\n")

        # Removing selected nodes and edges
        for node_id in pedestrian_nodes_to_remove:
            if node_id in nodes_in_xml:
                node_root.remove(nodes_in_xml[node_id]) # remove from nod component file
                del nodes_in_xml[node_id] # remove from dictionary

        for (f, t) in pedestrian_edges_to_remove:
            if (f, t) in edges_in_xml:
                edge_root.remove(edges_in_xml[(f, t)]) # remove from edg component file
                del edges_in_xml[(f, t)] # remove from dictionary

        # Before new nodes are added.
        # All the nodes with tl other than default tl need to have type="dead_end" and tl attribute removed.
        default_tl = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # By default in base, there will the one TL at the left intersection. present.
        for node in node_root.findall('node'):
            tl_name = node.get('tl')
            if tl_name:
                if tl_name not in default_tl:
                    node.set('type', 'dead_end')
                    del node.attrib['tl']

        # Find the pedestrian nodes to add (present in networkx graph but not in XML component file) i.e., end nodes and middle nodes
        # In iterations other than base i.e., in iteration base, there will be no new nodes to add.
        # For regular nodes: <node id=" " x=" " y=" " />
        # For the nodes with type "middle": also add attributes: type = "traffic_light" and tl = "node_id"
        node_ids_to_add = pedestrian_nodes_in_graph - set(nodes_in_xml.keys())
        middle_nodes_to_add = []
        print(f"\nDEBUG_XML: Nodes to add: {node_ids_to_add}")

        for nid in node_ids_to_add:
            node_data = networkx_graph.nodes[nid]
            x, y = node_data['pos']
            n_type = node_data.get('type', 'regular')
            attribs = {'id': nid, 'x': str(round(x, 2)), 'y': str(round(y, 2))}

            # --- DEBUG PRINT ---
            print(f"DEBUG_XML_NODE_ADD: Processing node ID: {nid}")
            print(f"  -> Node data: {node_data}")
            print(f"  -> x={x}, y={y}, type={n_type}")
            # --- END DEBUG ---

            if n_type == 'regular':
                attribs['type'] = 'dead_end'
            elif n_type == 'middle':
                middle_nodes_to_add.append(nid)
                attribs['type'] = 'traffic_light'
                attribs['tl'] = nid # Make sure nid is a string here (it should be from graph)

            # --- DEBUG PRINT ---
            print(f"  -> Final node attributes for ET.Element: {attribs}")
            # Check types before creating element
            for key, value in attribs.items():
                if not isinstance(value, str):
                    print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
            # --- END DEBUG ---

            new_node = ET.Element('node', attribs)
            new_node.tail = "\n\t"
            node_root.append(new_node)
            nodes_in_xml[nid] = new_node

        # Find the edges to add (present in networkx graph but not in XML component file).
        ped_edges_to_add = set(networkx_graph.edges()) - set(edges_in_xml.keys()) # These are all pedestrian edges.
        ped_edges_to_add = list(ped_edges_to_add)
        # print(f"\nPedestrian edges to add: Total: {len(ped_edges_to_add)},\n {ped_edges_to_add}\n")

        # The edge could be from a type = "regular" node to a type = "regular" node or from a type = "regular" node to a type = "middle" node (crossing).
        for (f, t) in ped_edges_to_add:
            # Do Regular to Regular and Regular to Middle need some different treatment?
            edge_data = networkx_graph.get_edge_data(f, t)
            edge_id = edge_data.get('id', f'edge_{f}_{t}') # Get it from the networkx graph.
            width = edge_data.get('width', None) # There should be a width for all edges.

            # --- DEBUG PRINT ---
            print(f"DEBUG_XML_PED_EDGE_ADD: Processing edge: ({f}, {t})")
            print(f"  -> Edge data from graph: {edge_data}")
            print(f"  -> Extracted edge_id={edge_id}, width={width} (Type: {type(width)})")
            if width is None:
                print("  -> WARNING: width is None, using default string '2.0' for XML.")
                str_width = '2.0' # Provide a default if None
            else:
                str_width = str(width)
            # --- END DEBUG ---

            edge_attribs = {
                'id': edge_id, # Should be string
                'from': f,     # Should be string (node ID)
                'to': t,       # Should be string (node ID)
                'name': 'Iterative addition',
                'priority': '1',
                'type': 'highway.footway',
                'numLanes': '1',
                'speed': '2.78', # default
                'spreadType': 'center',
                'width': str_width, # Use the safe string width
                'allow': 'pedestrian'
            }

            # --- DEBUG PRINT ---
            print(f"  -> Final edge attributes for ET.Element: {edge_attribs}")
            # Check types before creating element
            for key, value in edge_attribs.items():
                if not isinstance(value, str):
                    print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
            # --- END DEBUG ---

            # positions of f and t nodes
            f_data = networkx_graph.nodes[f]
            t_data = networkx_graph.nodes[t]
            f_x, f_y = round(f_data['pos'][0], 2), round(f_data['pos'][1], 2)
            t_x, t_y = round(t_data['pos'][0], 2), round(t_data['pos'][1], 2)
            shape = f'{f_x},{f_y} {t_x},{t_y}' # shape is inherently a string

            # --- DEBUG PRINT ---
            print(f"  -> Lane shape: {shape}, Lane width: {str_width}")
            # --- END DEBUG ---

            edge_element = ET.Element('edge', edge_attribs)
            edge_element.text = "\n\t\t"

            lane_attribs = { # Attributes for the lane element
                'index':'0',
                'allow':'pedestrian',
                'width': str_width, # Use safe string width again
                'speed':'2.78',
                'shape':shape
            }

            # --- DEBUG PRINT ---
            print(f"  -> Lane attributes for ET.SubElement: {lane_attribs}")
            # Check types before creating element
            for key, value in lane_attribs.items():
                if not isinstance(value, str):
                    print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
            # --- END DEBUG ---

            lane_element = ET.SubElement(edge_element, 'lane', **lane_attribs) # Use dictionary unpacking
            lane_element.text = "\n\t\t\t"

            param_element = ET.SubElement(lane_element, 'param', key='origId', value=edge_id) # edge_id should be string
            param_element.tail = "\n\t\t"
            lane_element.tail = "\n\t"
            edge_element.tail = "\n\t"
            edge_root.append(edge_element)

        # Every middle node (present in middle_nodes_to_add) falls on a certain vehicle edge. Split the vehicle edges into two new edges.
        # The new edge names have left and right attached to the old names (the new edges inherit respective portions of the edge shape and lane shape property of the old edge)
        # This happens iteratively (because multiple middle nodes may fall on the same vehicle edge) and is a bit complex.
        # NOTE: We pass connection_root here, which might be modified inside get_new_veh_edges_connections
        print(f"DEBUG_XML: Calling get_new_veh_edges_connections with middle_nodes_to_add: {middle_nodes_to_add}")
        old_veh_edges_to_remove, new_veh_edges_to_add, updated_conn_root, m_node_mapping = get_new_veh_edges_connections(middle_nodes_to_add,
                                                                                                            networkx_graph,
                                                                                                            f'{self.component_dir}/original.edg.xml',
                                                                                                            f'{self.component_dir}/original.nod.xml',
                                                                                                            connection_root) # Pass the original root
        # --- DEBUG PRINT ---
        print(f"DEBUG_XML_VEH_SPLIT: Results from get_new_veh_edges_connections:")
        print(f"  -> old_veh_edges_to_remove: {old_veh_edges_to_remove}")
        # Printing the full new_veh_edges_to_add and m_node_mapping might be verbose, print summary or specific problematic parts if needed
        print(f"  -> new_veh_edges_to_add keys: {new_veh_edges_to_add.keys() if new_veh_edges_to_add else 'None'}")
        print(f"  -> m_node_mapping keys: {m_node_mapping.keys() if m_node_mapping else 'None'}")
        print(f"  -> connection_root was potentially updated in place. Current root tag: {updated_conn_root.tag if updated_conn_root is not None else 'None'}")
        # --- END DEBUG ---

        # It seems 'updated_conn_root' might be the new root object for the connection tree
        # Let's assume connection_tree's root should now be updated if the function returned a new one
        if updated_conn_root is not None and updated_conn_root is not connection_root:
            print("DEBUG_XML: Updating connection_tree root with the one returned from get_new_veh_edges_connections.")
            connection_tree._setroot(updated_conn_root)
            connection_root = updated_conn_root # Update local reference too
        elif updated_conn_root is None:
            print("WARNING: get_new_veh_edges_connections returned None for updated_conn_root")
            # Handle this case - maybe raise error or use original connection_root cautiously

        # print(f"old_veh_edges_to_remove: {old_veh_edges_to_remove}\n")
        # print(f"new_veh_edges_to_add: {new_veh_edges_to_add}\n")

        # Add the new edges (each edge has a single nested lane) to the edge file. The width is the default road width.
        for direction in ['top', 'bottom']:
            if direction in new_veh_edges_to_add: # Check if direction exists
                for edge_id, edge_data in new_veh_edges_to_add[direction].items():
                    # --- DEBUG PRINT ---
                    print(f"DEBUG_XML_VEH_EDGE_ADD: Processing new vehicle edge: {edge_id} (Direction: {direction})")
                    print(f"  -> Edge data: {edge_data}")
                    if edge_data is None:
                        print("   -> ERROR: edge_data is None. Skipping.")
                        continue
                    # --- END DEBUG ---

                    edge_attribs = {
                        'id': edge_id,
                        'from': edge_data.get('from'), # Use .get for safety
                        'to': edge_data.get('to'),     # Use .get for safety
                        'name': "Craver Road Iterative Addition",
                        'priority': "10",
                        'type': "highway.tertiary",
                        'numLanes': "1",
                        'speed': "8.94",
                        'disallow': "pedestrian tram rail_urban rail rail_electric rail_fast ship cable_car subway"
                    }

                    # --- DEBUG PRINT ---
                    print(f"  -> Final edge attributes for ET.Element: {edge_attribs}")
                    # Check types before creating element
                    for key, value in edge_attribs.items():
                        if value is None:
                            print(f"    -> WARNING: Attribute '{key}' has None value!")
                        elif not isinstance(value, str):
                            print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                    # --- END DEBUG ---

                    edge_element = ET.Element('edge', edge_attribs)
                    edge_element.text = "\n\t\t"

                    lane_attribs = { # Attributes for the lane element
                        'index':'0',
                        'disallow':"pedestrian tram rail_urban rail rail_electric rail_fast ship cable_car subway",
                        'speed':"8.94",
                    }

                    # --- DEBUG PRINT ---
                    print(f"  -> Lane attributes for ET.SubElement: {lane_attribs}")
                    # Check types before creating element
                    for key, value in lane_attribs.items():
                        if not isinstance(value, str):
                            print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                    # --- END DEBUG ---

                    lane_element = ET.SubElement(edge_element, 'lane', **lane_attribs)

                    lane_element.text = "\n\t\t\t"
                    param_val = edge_id.split('#')[0].replace('-', '') # Should be string

                    # --- DEBUG PRINT ---
                    print(f"  -> Param element value: {param_val}")
                    # --- END DEBUG ---

                    param_element = ET.SubElement(lane_element, 'param', key='origId', value=param_val)
                    param_element.tail = "\n\t\t"
                    lane_element.tail = "\n\t"
                    edge_element.tail = "\n\t"

                    edge_root.append(edge_element)
            else:
                print(f"DEBUG_XML_VEH_EDGE_ADD: No new vehicle edges to add for direction '{direction}'.")


        # For TL logics,
        # TL logics should come before the connections. (https://github.com/eclipse-sumo/sumo/issues/6160)
        # In order to do this, we first remove all existing TL logics except the default one.
        # We collect the connections associated with default TL and remove all connections.
        # TL 1. Remove all TLs and except the default one.
        tls_to_remove = []
        for tl in traffic_light_root.findall('tlLogic'):
            if tl.get('id') not in default_tl:
                tls_to_remove.append(tl)
        for tl in tls_to_remove:
            traffic_light_root.remove(tl)

        # TL 2. Remove all connections and store the default ones.
        tl_connections_to_add = [] # collect the connection elements.
        connections_to_remove_from_con_file = [] # Connections except the default TL should be removed from the connections file as well.
        print(f"DEBUG_XML_TL: Processing connections within TLL file...")
        for conn in traffic_light_root.findall('connection'):
            tl_id_for_conn = conn.get('tl')
            print(f"  -> Found TLL connection from={conn.get('from')} to={conn.get('to')} for tl={tl_id_for_conn}")
            traffic_light_root.remove(conn) # remove from the TLL file whether its default or not. We will add it back later.
            if tl_id_for_conn in default_tl:
                print(f"    -> Keeping (to re-add later) as it belongs to default TL.")
                tl_connections_to_add.append(conn)
            else:
                print(f"    -> Marking for removal from .con file.")
                # Store the element itself or its identifying attributes
                connections_to_remove_from_con_file.append({'from': conn.get('from'), 'to': conn.get('to'), 'fromLane': conn.get('fromLane'), 'toLane': conn.get('toLane')})


        # The TLL file connections contains connections between edges that are left and right of every midde node.
        # Due to split of split, the names of these edges may not be symmetrical (i.e., just replace left with right and vice versa wont work).
        # Use linkIndex 0 for connecting -ve direction and linkIndex 1 for connecting +ve direction.
        print(f"DEBUG_XML_TL: Creating new TL and regular connections based on m_node_mapping...")
        for direction in ['top', 'bottom']:
            if m_node_mapping: # Check if mapping exists
                for tl_id, mapping_data in m_node_mapping.items(): # m_node is the tl_id
                    linkindex = 0 if direction == 'top' else 1 # Top is -ve direction and bottom is +ve direction.
                    print(f"DEBUG_XML_TL_CONN_ADD: Processing TL connection for tl_id={tl_id}, direction={direction}")
                    print(f"  -> Mapping data for direction: {mapping_data.get(direction, 'MISSING')}") # Check if key exists

                    if mapping_data and direction in mapping_data: # Check before accessing
                        map_dir_data = mapping_data[direction]
                        from_edge = map_dir_data.get('from')
                        to_edge = map_dir_data.get('to')

                        # --- DEBUG PRINT ---
                        print(f"  -> Extracted from_edge='{from_edge}', to_edge='{to_edge}' for tl_id='{tl_id}'")
                        if from_edge is None or to_edge is None:
                            print("    -> ERROR: from_edge or to_edge is None in mapping_data. Skipping connection creation.")
                            continue
                        # --- END DEBUG ---

                        # These connections should be present in both the TLL and connections files (using left as from and right as to).
                        # TL 3. Add the new connections.
                        tl_conn_attribs = {'from': from_edge, 'to': to_edge, 'fromLane': "0", 'toLane': "0", 'tl': tl_id, 'linkIndex': str(linkindex)} # Since inside the corridor, there is only one lane.

                        # --- DEBUG PRINT ---
                        print(f"  -> TL Connection attributes for TLL file: {tl_conn_attribs}")
                        # Check types before creating element
                        for key, value in tl_conn_attribs.items():
                            if not isinstance(value, str):
                                print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                        # --- END DEBUG ---

                        tl_conn_element = ET.Element('connection', tl_conn_attribs)
                        tl_connections_to_add.append(tl_conn_element) # Add to list for later insertion into TLL

                        conn_attribs = {'from': from_edge, 'to': to_edge, 'fromLane': "0", 'toLane': "0"} # Since inside the corridor, there is only one lane.

                        # --- DEBUG PRINT ---
                        print(f"  -> Regular Connection attributes for CON file: {conn_attribs}")
                        # Check types before creating element
                        for key, value in conn_attribs.items():
                            if not isinstance(value, str):
                                print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                        # --- END DEBUG ---

                        conn_element = ET.Element('connection', conn_attribs)
                        conn_element.text = None  # Ensure there's no text content
                        conn_element.tail = "\n\t" # Adjusted tail for .con file format
                        if connection_root is not None:
                            connection_root.append(conn_element) # Append directly to connection root
                        else:
                            print("   -> ERROR: connection_root is None, cannot append connection element!")

                    else:
                        print(f"  -> WARNING: Missing mapping data or direction '{direction}' in tl_id '{tl_id}'")
            else:
                print(f"DEBUG_XML_TL_CONN_ADD: m_node_mapping is empty or None. Skipping connection creation.")


        # For the crossing tags in the Conn file ( which also dont need to be changed iteratively). # The width here needs to come from the model.
        # They are already updated while obtaining the new edges. Nothing to do here? Check get_new_veh_edges_connections.
        # Assumes get_new_veh_edges_connections potentially modified connection_root in place or returned the updated one.

        # Whereas for the crossing tags,
        # First remove all except the default ones. Then add the new ones here by making use of new_veh_edges_to_add.
        default_crossings = default_tl + ['cluster_172228408_9739966907_9739966910', '9687187500', '9687187501'] # associated with ids 0 and 10.
        print(f"DEBUG_XML_CROSSING: Removing non-default crossings from connection root...")
        if connection_root is not None:
            crossings_to_remove = []
            for crossing in connection_root.findall('crossing'):
                node_id = crossing.get('node')
                print(f"  -> Found crossing for node: {node_id}")
                if node_id not in default_crossings:
                    print(f"    -> Marking for removal (not in default list: {default_crossings})")
                    crossings_to_remove.append(crossing)
            for crossing in crossings_to_remove:
                connection_root.remove(crossing)
        else:
            print("   -> ERROR: connection_root is None, cannot remove crossings!")


        # Then deal with the existing old crossings that refer to the old edges which have been split.
        # Can be done manually.. as in -> if the leftmost edge has been split then the intersection should now refer to the new edge.
        min_x, max_x = float('inf'), float('-inf')
        leftmost_new, rightmost_new = '', ''
        if 'top' in new_veh_edges_to_add: # Check if 'top' exists
            for edge_id, edge_data in new_veh_edges_to_add['top'].items(): # One of the counterparts (among -ve, +ve) is enough.
                if edge_data: # Check if edge_data is not None
                    # Also bottom has reverse direction so top is enough.
                    from_x = edge_data.get('from_x', float('inf')) # Default to prevent error if missing
                    to_x = edge_data.get('to_x', float('inf'))     # Default to prevent error if missing
                    min_x_among_nodes = min(from_x, to_x)

                    if min_x_among_nodes < min_x:
                        min_x = min_x_among_nodes
                        # Ensure edge_id format allows splitting and forming the positive counterpart
                        if '#' in edge_id:
                            leftmost_new = f'16666012#{edge_id.split("#")[1]}'
                        else:
                            print(f"WARNING: Edge ID format unexpected for leftmost edge update: {edge_id}")


                    if min_x_among_nodes > max_x and min_x_among_nodes != float('inf'): # Check against default
                        max_x = min_x_among_nodes
                        # Ensure edge_id format allows splitting and forming the positive counterpart
                        if '#' in edge_id:
                            rightmost_new = f'16666012#{edge_id.split("#")[1]}'
                        else:
                            print(f"WARNING: Edge ID format unexpected for rightmost edge update: {edge_id}")
                else:
                    print(f"WARNING: edge_data is None for edge_id {edge_id} in new_veh_edges_to_add['top']")

        # One of the counterparts (among -ve, +ve) is enough.
        self.extreme_edge_dict['leftmost']['new'] = leftmost_new
        self.extreme_edge_dict['rightmost']['new'] = rightmost_new
        print(f"DEBUG_XML_EXTREME: Updated extreme_edge_dict: {self.extreme_edge_dict}")


        # Updates to connections and crossings in connections file.
        print(f"DEBUG_XML_EXTREME: Updating connections/crossings for extreme edges...")
        if connection_root is not None:
            for direction, direction_data in self.extreme_edge_dict.items():
                old_edge = direction_data['old']
                if old_edge in old_veh_edges_to_remove:
                    new_edge = direction_data['new']
                    print(f"DEBUG_XML_EXTREME_UPDATE: Updating for direction '{direction}'. Old: {old_edge}, New: {new_edge}")
                    if not new_edge: # Check if new_edge is empty or None
                        print(f"  -> WARNING: new_edge for {direction} is empty. Skipping updates for this direction.")
                        continue

                    crossings_to_update = connection_root.findall('crossing')
                    connections_to_add_for_crossing = []

                    for crossing in crossings_to_update:
                        current_edges = crossing.get('edges')
                        print(f"  -> Checking crossing with edges: '{current_edges}'")
                        updated = False
                        new_crossing_edges = None

                        if current_edges == f'{old_edge} -{old_edge}':
                            # First, prepare a connection between the two new edges to be added.
                            conn_attrs = {'from': new_edge, 'to': f'-{new_edge}', 'fromLane': '0', 'toLane': '0'}
                            print(f"    -> Matched {old_edge} -{old_edge}. Preparing connection with attrs: {conn_attrs}")
                            connections_to_add_for_crossing.append(conn_attrs)
                            # Then, it can be updated in crossing.
                            new_crossing_edges = f'{new_edge} -{new_edge}'
                            updated = True

                        elif current_edges == f'-{old_edge} {old_edge}':
                            # First, prepare a connection between the two new edges to be added.
                            conn_attrs = {'from': f'-{new_edge}', 'to': new_edge, 'fromLane': '0', 'toLane': '0'}
                            print(f"    -> Matched -{old_edge} {old_edge}. Preparing connection with attrs: {conn_attrs}")
                            connections_to_add_for_crossing.append(conn_attrs)
                            # Then, it can be updated in crossing.
                            new_crossing_edges = f'-{new_edge} {new_edge}'
                            updated = True

                        if updated:
                            print(f"    -> Setting crossing edges to: {new_crossing_edges}")
                            # --- DEBUG PRINT ---
                            # Check types before setting attribute
                            if not isinstance(new_crossing_edges, str):
                                print(f"      -> WARNING: new_crossing_edges is not a string: {new_crossing_edges} (Type: {type(new_crossing_edges)})")
                            # --- END DEBUG ---
                            crossing.set('edges', new_crossing_edges)

                    # Add the prepared connections after iterating through crossings
                    for conn_attrs in connections_to_add_for_crossing:
                        print(f"  -> Adding connection element to connection_root with attrs: {conn_attrs}")
                        # --- DEBUG PRINT ---
                        # Check types before creating element
                        for key, value in conn_attrs.items():
                            if not isinstance(value, str):
                                print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                        # --- END DEBUG ---
                        connection_element = ET.Element('connection', conn_attrs)
                        connection_element.text = None
                        connection_element.tail = "\n\t" # Check if tail is correct for .con file
                        connection_root.append(connection_element)

        else:
            print("   -> ERROR: connection_root is None, cannot update extreme edge connections/crossings!")


        # Add new connections (between top and bottom edges) and crossings (making use of new_veh_edges_to_add).
        # All tags that refer to the old edges should now refer to the new edges (if the refering edges fall to the left, they will refer to the new left edge and vice versa)
        # They have the edges attribute (which are edges to the right) and outlineShape attribute (the shape of the crossing):

        # outlineShape seems hard to specify, lets not specify and see what it does. They mention it as optional here: https://github.com/eclipse-sumo/sumo/issues/11668
        # TODO: same node contains right and left components which creates two crossings instead of one. Find a way to avoid this (Only add the right part of the crossing).
        print(f"DEBUG_XML_CROSSING: Adding new crossings based on new vehicle edges...")
        if connection_root is not None and 'top' in new_veh_edges_to_add: # Check root and key exist
            for e1, e1_data in new_veh_edges_to_add['top'].items(): # Just looking at one direction (top) is enough.
                if e1_data and 'right' in e1.split('_')[-1]: # Add only the right part & check e1_data
                    e2 = e1.replace('-', '') # To get the bottom edge id.
                    print(f"DEBUG_XML_CROSSING_ADD: Processing new crossing based on right-side edge e1={e1}, corresponding e2={e2}")

                    middle_node = e1_data.get('new_node')
                    print(f"  -> Corresponding middle_node: {middle_node} (Type: {type(middle_node)})")

                    # IMPORTANT: Check if middle_node itself is None before using it as key
                    if middle_node is None:
                        print("  -> ERROR: middle_node is None, cannot create crossing!")
                        continue # Skip this crossing

                    # Safely get width from graph
                    width = None
                    if middle_node in networkx_graph.nodes:
                        width = networkx_graph.nodes[middle_node].get('width')
                    else:
                        print(f"  -> WARNING: middle_node '{middle_node}' not found in networkx_graph. Cannot get width.")

                    print(f"  -> Corresponding width: {width} (Type: {type(width)})")

                    # Make width a string, handle None or missing node case explicitly
                    str_width = "2.0" # Default width if missing or None
                    if width is not None:
                        str_width = str(width)
                    else:
                        print(f"  -> Using default width '{str_width}' for crossing.")

                    crossing_attribs = {
                        'node': middle_node, # middle_node is checked for None above
                        'edges': e1 + ' ' + e2, # e1, e2 should be strings
                        'priority': '1',
                        'width': str_width, # Use safe string width
                        'linkIndex': '2'
                    }

                    # --- DEBUG PRINT ---
                    print(f"  -> Final crossing attributes for ET.Element: {crossing_attribs}")
                    # Check types before creating element
                    for key, value in crossing_attribs.items():
                        if not isinstance(value, str):
                            print(f"    -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                    # --- END DEBUG ---

                    crossing_element = ET.Element('crossing', crossing_attribs)
                    crossing_element.text = None  # Ensure there's no text content
                    crossing_element.tail = "\n\t" # Adjusted tail for .con file
                    connection_root.append(crossing_element)
                elif not e1_data:
                    print(f"DEBUG_XML_CROSSING_ADD: Skipping edge e1={e1} because its data is None.")

        elif connection_root is None:
            print("   -> ERROR: connection_root is None, cannot add new crossings!")
        elif 'top' not in new_veh_edges_to_add:
            print("   -> INFO: 'top' key not found in new_veh_edges_to_add, cannot add new crossings.")


        # Delete the old edges from the edg file i.e., just remove the tags with old edge ids.
        print(f"DEBUG_XML: Removing old vehicle edges from edge root...")
        edges_to_remove_from_root = []
        for edge in edge_root.findall('edge'):
            edge_id = edge.get('id')
            if edge_id in old_veh_edges_to_remove:
                print(f"  -> Marking edge '{edge_id}' for removal.")
                edges_to_remove_from_root.append(edge)
        for edge in edges_to_remove_from_root:
            edge_root.remove(edge)

        # TL 4. Add the new TL logics.
        print(f"DEBUG_XML_TL: Adding new TL Logic elements...")
        for nid in middle_nodes_to_add:
            # --- DEBUG PRINT ---
            print(f"DEBUG_XML_TL_LOGIC_ADD: Adding TL Logic for node ID: {nid}")
            if not isinstance(nid, str):
                print(f"  -> WARNING: Node ID (TL ID) is not a string: {nid} (Type: {type(nid)})")
            # --- END DEBUG ---
            tlLogic_element = ET.Element('tlLogic', id=nid, type='static', programID='0', offset='0')
            tlLogic_element.text = "\n\t\t" # Inside <tlLogic>: phases start at two tabs

            # Create phases with proper indentation
            phase1 = ET.SubElement(tlLogic_element, 'phase', duration='77', state='GGr')
            phase1.tail = "\n\t\t"
            phase2 = ET.SubElement(tlLogic_element, 'phase', duration='3', state='yyr')
            phase2.tail = "\n\t\t"
            phase3 = ET.SubElement(tlLogic_element, 'phase', duration='5', state='rrG')
            phase3.tail = "\n\t\t"
            phase4 = ET.SubElement(tlLogic_element, 'phase', duration='5', state='rrr')
            phase4.tail = "\n\t" # Tail for the last phase before closing tag

            tlLogic_element.tail = "\n\t" # Tail for the tlLogic element itself
            traffic_light_root.append(tlLogic_element)

        # TL 5. Add all the new and default connections back to the TLL file.
        print(f"DEBUG_XML_TL: Re-adding collected TL connection elements to TLL root...")
        for conn in tl_connections_to_add:
            # --- DEBUG PRINT ---
            print(f"DEBUG_XML_TL_CONN_RE_ADD: Re-adding TL connection element with attributes: {conn.attrib}")
            # Check types before adding element back (should be okay if checked during creation)
            for key, value in conn.attrib.items():
                if not isinstance(value, str):
                    print(f"  -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
            # --- END DEBUG ---
            conn.text = None
            conn.tail = "\n\t"
            traffic_light_root.append(conn)

        # TL 6. The default crossings in TL (that were kept above) may still refer to the old edges.
        # In addition, there may also be a connection of the -ve and +ve sides of the old edges.
        print(f"DEBUG_XML_TL: Updating default TL connections if their edges were split...")
        for direction, direction_data in self.extreme_edge_dict.items():
            old_edge = direction_data['old']
            if old_edge in old_veh_edges_to_remove:
                new_edge = direction_data['new']
                print(f"DEBUG_XML_DEFAULT_TL_CONN_UPDATE: Updating default TL connections for Old: {old_edge}, New: {new_edge}")
                if not new_edge:
                    print(f"  -> WARNING: new_edge for {direction} is empty. Skipping update for default TL connections.")
                    continue

                for conn in traffic_light_root.findall('connection'):
                    print(f"  -> Checking TL connection element with attributes: {conn.attrib}") # Before modification
                    modified = False
                    current_from = conn.get('from')
                    current_to = conn.get('to')

                    if current_from == old_edge: # positive
                        print(f"    -> Updating 'from' from {old_edge} to {new_edge}")
                        conn.set('from', new_edge)
                        modified = True
                    if current_from == f"-{old_edge}": # negative
                        print(f"    -> Updating 'from' from -{old_edge} to -{new_edge}")
                        conn.set('from', f"-{new_edge}")
                        modified = True
                    if current_to == old_edge: # positive
                        print(f"    -> Updating 'to' from {old_edge} to {new_edge}")
                        conn.set('to', new_edge)
                        modified = True
                    if current_to == f"-{old_edge}": # negative
                        print(f"    -> Updating 'to' from -{old_edge} to -{new_edge}")
                        conn.set('to', f"-{new_edge}")
                        modified = True

                    if modified:
                        print(f"    --> Modified TL connection attributes: {conn.attrib}") # After modification
                        # --- DEBUG PRINT ---
                        # Check types after modification
                        for key, value in conn.attrib.items():
                            if not isinstance(value, str):
                                print(f"      -> WARNING: Attribute '{key}' has non-string value: {value} (Type: {type(value)})")
                        # --- END DEBUG ---


        # Respective changes to the connections file.
        # All the connections present in the TLL file should also be present in the connections file. But the connection file will have more of them.
        # In iteration base, there will be a bunch of connections to remove from original file (remove connections with the same from and to edges).
        # all_conn_file_connections = [(conn.get('from'), conn.get('to')) for conn in connection_root.findall('connection')]
        # print(f"connection Before removal: Total: {len(all_conn_file_connections)},\n {all_conn_file_connections}\n")

        # Look at the same from and to edges in the connections file and remove them.
        # These were connections associated with non-default TLs in the original TLL file.
        print(f"DEBUG_XML_CON: Removing connections from .con file that belonged to removed TLs...")
        if connection_root is not None:
            conns_to_remove_from_con_root = []
            for conn_element in connection_root.findall('connection'):
                f = conn_element.get('from')
                t = conn_element.get('to')
                fl = conn_element.get('fromLane')
                tlane = conn_element.get('toLane')
                # Check if this connection matches one marked for removal
                for removal_marker in connections_to_remove_from_con_file:
                    if (f == removal_marker['from'] and
                        t == removal_marker['to'] and
                        fl == removal_marker['fromLane'] and
                        tlane == removal_marker['toLane']):
                        print(f"  -> Marking connection from={f} to={t} lanes={fl}->{tlane} for removal from .con file.")
                        conns_to_remove_from_con_root.append(conn_element)
                        break # Move to next connection element once matched

            for conn_element in conns_to_remove_from_con_root:
                connection_root.remove(conn_element)
        else:
            print("   -> ERROR: connection_root is None, cannot remove connections!")

        # Additional stuff related to edge removals.
        # If the edge (pedestrian and vehicle) is removed, then the connections to and from that edge should also be removed.
        pedestrian_edges_ids_to_remove = []
        for (f,t), edge in pedestrian_edges_to_remove.items():
            edge_id = edge.get('id')
            if edge_id:
                pedestrian_edges_ids_to_remove.append(edge_id)

        print(f"DEBUG_XML_CON: Removing connections involving removed pedestrian edges: {pedestrian_edges_ids_to_remove}")

        if connection_root is not None:
            conns_involving_removed_edges = []
            for conn in connection_root.findall('connection'):
                f = conn.get('from')
                t = conn.get('to')
                if f in pedestrian_edges_ids_to_remove or t in pedestrian_edges_ids_to_remove:
                    print(f"  -> Marking connection from={f} to={t} for removal (involves removed ped edge).")
                    conns_involving_removed_edges.append(conn)
            for conn in conns_involving_removed_edges:
                connection_root.remove(conn)
        else:
            print("   -> ERROR: connection_root is None, cannot remove connections involving removed ped edges!")


        iteration_prefix = f'{self.component_dir}/iteration_{iteration}'
        print(f"\nDEBUG_XML: Writing final component XML files with prefix: {iteration_prefix}")
        node_tree.write(f'{iteration_prefix}.nod.xml', encoding='utf-8', xml_declaration=True)
        edge_tree.write(f'{iteration_prefix}.edg.xml', encoding='utf-8', xml_declaration=True)
        # Use the potentially updated connection_tree (check if its root was replaced)
        connection_tree.write(f'{iteration_prefix}.con.xml', encoding='utf-8', xml_declaration=True)
        type_tree.write(f'{iteration_prefix}.typ.xml', encoding='utf-8', xml_declaration=True)
        traffic_light_tree.write(f'{iteration_prefix}.tll.xml', encoding='utf-8', xml_declaration=True)
        print("DEBUG_XML: Component files written.")


        # Generate the final net file using netconvert
        output_file = f'{self.network_dir}/network_iteration_{iteration}.net.xml'
        netconvert_log_file = f'simulation/netconvert_log_{iteration}.txt' # Unique log per iteration
        command = (
            f"netconvert "
            f"--node-files={iteration_prefix}.nod.xml "
            f"--edge-files={iteration_prefix}.edg.xml "
            f"--connection-files={iteration_prefix}.con.xml "
            f"--type-files={iteration_prefix}.typ.xml "
            f"--output-file={output_file} "
            f"--log={netconvert_log_file} "
            f"--xml-validation never " # Temporarily disable strict validation if needed for debugging netconvert errors
            f"--verbose" # Get more output from netconvert
        )
        print(f"DEBUG_XML: Running netconvert command:\n{command}")
        # If you want to use the tlogic file, place after type file in the command above
        # f"--tllogic-files={iteration_prefix}.tll.xml "

        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                # Using PIPE for stdout/stderr allows capturing them if needed
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=30) # Added timeout
                print(f"Netconvert stdout (attempt {attempt+1}):\n{result.stdout}")
                if result.stderr:
                    # Warnings are often printed to stderr by netconvert
                    print(f"Netconvert stderr/warnings (attempt {attempt+1}):\n{result.stderr}")
                print(f"Netconvert finished successfully on attempt {attempt+1}.")
                break # Success
            except subprocess.CalledProcessError as e:
                print(f"Error running netconvert (attempt {attempt + 1}/{max_attempts}): {e}")
                print(f"Netconvert return code: {e.returncode}")
                print(f"Netconvert stdout:\n{e.stdout}")
                print(f"Netconvert stderr:\n{e.stderr}")
                attempt += 1
                if attempt == max_attempts:
                    print("Failed all attempts to run netconvert. Check logs and XML files.")
                    print(f"Check netconvert log: {netconvert_log_file}")
                    print(f"Check component files prefixed with: {iteration_prefix}")
                    # Optionally: Copy problematic files for inspection
                    # shutil.copy(f'{iteration_prefix}.nod.xml', f'{iteration_prefix}.nod.xml.failed')
                    # ... etc for other files ...
                    raise # Re-raise the exception after logging
            except subprocess.TimeoutExpired as e:
                print(f"Netconvert timed out (attempt {attempt + 1}/{max_attempts}): {e}")
                print(f"Netconvert stdout:\n{e.stdout}")
                print(f"Netconvert stderr:\n{e.stderr}")
                attempt += 1
                if attempt == max_attempts:
                    print("Failed all attempts to run netconvert due to timeout.")
                    raise