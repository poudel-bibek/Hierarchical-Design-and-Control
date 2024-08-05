def _disallow_pedestrians(self, all_crosswalks, crosswalks_to_disable):
    """        
    """

    # Based on the timesteps, allow a bunch and then disallow the bunch
    print(f"Step count: {self.step_count}") # Increments by 10 here

    # Define time ranges for disabling crosswalks
    time_ranges = [
        (200, 500, all_crosswalks[:5]),
        (500, 1000, all_crosswalks[5:]),
        (1000, float('inf'), all_crosswalks[:5])
    ]

    # Determine which crosswalks to disable based on current step
    crosswalks_to_disable = [] 
    for start, end, crosswalks in time_ranges:
        if start < self.step_count <= end:
            print(f"\nTime range: {start} - {end}, Disabled Crosswalks: {crosswalks}")
            crosswalks_to_disable = crosswalks
            break
        
    disabled_crosswalks = []

    walking_edges_to_disable = []
    for crosswalk_id in crosswalks_to_disable:
        walking_edges_to_disable.extend(self.crosswalk_to_walking_areas[crosswalk_id])
    print(f"\nWalking edges to disable: {walking_edges_to_disable}\n")

    walking_lanes_to_disable = [ edge + '_0' for edge in walking_edges_to_disable]
    print(f"\nWalking lanes to disable: {walking_lanes_to_disable}\n")

    # 2. Reroute pedestrians. This needs to happen every step as new pedestrians could spawn.
    pedestrian_ids = traci.person.getIDList()
    for ped_id in pedestrian_ids:

        current_edge = traci.person.getRoadID(ped_id)
        

        # print(f"Pedestrian ID: {ped_id}, Current Edge: {current_edge}")
        # Dont re-route if the current edge is already the crosswalk or an internal edge, it fails.

        if current_edge in walking_edges_to_disable: # If the pedestrian is in one of the edges that we want to disable

            # Find alternative crosswalks (walking areas)
            alternative_crosswalks = [
                crosswalk_data['ids']
                for crosswalk_data in self.controlled_crosswalks_dict.values()
                if not any(cw in crosswalks_to_disable for cw in crosswalk_data['ids'])
            ]
            # print(f"\nAlternative crosswalks: {alternative_crosswalks}\n")

            # Flatten the list of lists
            alternative_crosswalks = [
                item for sublist in alternative_crosswalks for item in sublist
            ]
            # print(f"\nAlternative crosswalks flattened: {alternative_crosswalks}\n")

            # Get the remaining stages in the route
            remaining_stages_count = traci.person.getRemainingStages(ped_id)
            print(f"\nRemaining stages count: {remaining_stages_count}\n")

            # Get the remaining edges in the person's route
            remaining_edges = []
            for i in range(remaining_stages_count):  # Start from 1 to get future stages
                stage = traci.person.getStage(ped_id, i)
                remaining_edges.extend(stage.edges)

            print(f"\nRemaining edges: {remaining_edges}\n")

            # If the person is about to use a disabled crosswalk (walking area), reroute
            if self._route_includes_disabled_crosswalk(remaining_edges, walking_lanes_to_disable):
                print(f"\nRerouting pedestrian {ped_id}\n")

                # When re-routing, first get the destination (end) edge in current stage
                # All this to get the destination edge
                # Get all the stages
                all_stages_count = traci.person.getRemainingStages(ped_id) 
                print(f"\nAll stages count: {all_stages_count}\n")

                route_stages = []
                for i in range(all_stages_count):
                    stage = traci.person.getStage(ped_id, i)
                    route_stages.append(stage)

                print(f"\nAll route stages: {route_stages}\n")

                # Pedestrians cant switch modes
                # Remove the current stage if the pedestrian is still in a walking area
                destination_edge = route_stages[-1].edges[-1]
                
                # Second, clear the remaining stages of the pedestrian
                # Automatically removes the person soon.
                traci.person.removeStages(ped_id)

                # current edge is a problem because it can contain internal edges, need to find re-route point for current edge.
                current_reroute_edges = self.all_edges_to_reroute_points.get(current_edge)
                upside_edge = current_reroute_edges.get('upside')
                downside_edge = current_reroute_edges.get('downside')
                
                # TODO: right now, I am just using upside. Need to find if the pedestrian currently is closer to upside or downside
                current_reroute_edge = upside_edge
                
                # Based on whether current is upside or downside, select the next one as downside or upside.
                current_direction = self.reroute_point_to_direction.get(current_reroute_edge) 

                # Third, get a new random crosswalk. TODO: Make this shortest path based
                new_crosswalk = random.choice(alternative_crosswalks)
                # Get the re-route point related to this new crosswalk
                next_reroute_edge = self.crosswalk_to_reroute_points[new_crosswalk].get(current_direction)
                
                print(f"\nCurrent edge: { current_reroute_edge}\n")
                print(f"\nNext Reroute point: {next_reroute_edge}\n")
                print(f"\nDestination edge: {destination_edge}\n")

                # There may still be a missing connection from the current edge to the current reroute edge
                # Move the person there, using coordinates

                # # Fourth, append two new walking stages:
                # #    - One from the current edge to the new crosswalk
                input_edges = [current_reroute_edge, next_reroute_edge]
                print(f"\nInput edges: {input_edges}\n")
                connecting_edges_1 = sumolib.route.addInternal(self.net, input_edges)
                # connecting_edges_1 = self.find_connecting_edges(current_reroute_edge, next_reroute_edge)

                print(f"\nConnecting edges 1: {connecting_edges_1}\n")
                traci.person.appendWalkingStage(ped_id, connecting_edges_1, 0) # Assume arrival position to be the end (0)
                
                # #    - Another from the new crosswalk to the destination edge
                input_edges = [next_reroute_edge, destination_edge]
                print(f"\nInput edges 2: {input_edges}\n")
                connecting_edges_2  = sumolib.route.addInternal(self.net, input_edges)
                # connecting_edges_2  = self.find_connecting_edges(next_reroute_edge, destination_edge)

                print(f"\nConnecting edges 2: {connecting_edges_2}\n")
                traci.person.appendWalkingStage(ped_id, connecting_edges_2, 0) # Assume arrival position to be the end (0)
                
                # # Print all stages again
                # all_stages_count = traci.person.getRemainingStages(ped_id)
                # print(f"\nAll stages count after rerouting: {all_stages_count}\n")
                
                #traci.person.remove(ped_id)

    #disabled_crosswalks.append(crosswalk_id)
    return disabled_crosswalks


def _route_includes_disabled_crosswalk(self, remaining_edges, walking_lanes_to_disable):
    """
    Check if a route (list of edges) includes any of the disabled crosswalks.
    """

    to_lanes = []
    # For all the remaining edges of this pedestrian's route
    for i in range(len(remaining_edges) - 1):
        from_edge = remaining_edges[i]
        
        # Returns descriptions of the links outgoing from this lane [m]
        # A list containing id of successor lane together with priority
        connecting_lanes = traci.lane.getLinks(from_edge + "_0")  # Assume first lane of the edge
        #print(f"\nConnecting lanes: {connecting_lanes}\n")

        for to_lane, _, _, _, _, _, _, _ in connecting_lanes:
            to_lanes.append(to_lane)

    print(f"\nTo lanes: {to_lanes}\n")
    if any(to_lane in walking_lanes_to_disable for to_lane in to_lanes):
        print(f"\nRoute includes disabled crosswalk\n")
        return True

    return False

#################################

    def _disallow_pedestrians(self, all_crosswalks, crosswalks_to_disable):
        """        
        """

        # Based on the timesteps, allow a bunch and then disallow the bunch
        print(f"Step count: {self.step_count}") # Increments by 10 here

        # Define time ranges for disabling crosswalks
        time_ranges = [
            (200, 500, all_crosswalks[:5]),
            (500, 1000, all_crosswalks[5:]),
            (1000, float('inf'), all_crosswalks[:5])
        ]

        # Determine which crosswalks to disable based on current step
        crosswalks_to_disable = [] # TODO: remove this. For test
        for start, end, crosswalks in time_ranges:
            if start < self.step_count <= end:
                print(f"\nTime range: {start} - {end}, Disabled Crosswalks: {crosswalks}")
                crosswalks_to_disable = crosswalks
                break
    
        #print(f"\nCrosswalks to disable: {crosswalks_to_disable}\n")
        
        for crosswalk_id in crosswalks_to_disable:

            all_associate_edges = [crosswalk_id] + list(set(self.all_edges_in_crosswalk[crosswalk_id])) # Set because it hs repititions
            print(f"\nAll associate edges: {all_associate_edges}\n")

            pedestrians = []
            for edge in all_associate_edges:

               

                # get all pedestrians on this edge
                persons = list(traci.edge.getLastStepPersonIDs(edge))
                if persons: # If not empty
                    pedestrians.extend(persons)

                #traci.edge.remove(edge)  # Remove the crosswalk edge

                # Either change the poperties of the edge.
                # traci.edge.setMaxSpeed(edge, 0.01) # Make it very slow TODO: Set speed back to normal
                # Use adapt travel time and set it to really high.
                #traci.edge.adaptTraveltime(edge, 500) # Set it to a really high value
                
                #new_time = traci.edge.getAdaptedTraveltime(edge, 50)
                #print(f"\nEdge: {edge}, New time: {new_time}\n")

                #traci.edge.setDisallowed(edge, ["pedestrian"]) # Red color

                # print(f"\nEdge: {edge}, Travel time: {traci.edge.getAdaptedTraveltime(edge, 50)}, Effort: {traci.edge.getEffort(edge, self.step_count)}\n")

                # # First make these edges difficult. 
                # traci.edge.setMaxSpeed(edge, 0.001)
                # traci.edge.adaptTraveltime(edge, 1500)
                # traci.edge.setEffort(edge, 10000) # A high value of effort means that the edge is difficult to traverse. More Costly

                # print(f"\nEdge: {edge}, Travel time: {traci.edge.getAdaptedTraveltime(edge, 50)}, Effort: {traci.edge.getEffort(edge, self.step_count)}\n")

            print(f"\nTotal: {len(pedestrians)}, Pedestrians: {pedestrians}\n")
            # If no person in all edges for this crosswalk
            # Direct approach
            # if len(pedestrians) == 0:
            #     traci.edge.setDisallowed(edge, ["pedestrian"])
                
            # Or do manually-reroute with the pedestrians
            # if pedestrians:
            #     # Now find a new route
            #     for person_id in pedestrians:
            #         traci.person.rerouteTraveltime(person_id)

            # 
            # if pedestrians:
            #     # Now find a new route
            #     for person_id in pedestrians:
                    
            #         # Find pedestrians remaining edges
            #         remaining_stages = traci.person.getRemainingStages(person_id)
            #         remaining_edges = [] 
            #         for stage in range(remaining_stages):
            #             remaining_edges.extend(traci.person.getEdges(person_id, stage)) # It actually gets all edges.

            #         print(f"\nPerson ID: {person_id}, Remaining edges: {remaining_edges}\n")

            #         # Find pedestrians current edge
            #         current_edge = remaining_edges[0] #traci.person.getRoadID(person_id)
            #         end_edge = remaining_edges[-1]

            #         print(f"\nCurrent edge: {current_edge}\n")

            #         #current_stage = traci.person.getEdges(person_id, 0)
            #         #print(f"\nCurrent stage: {current_stage}\n")
                    
            #         # The problem with this is that I get the same set of Edges.
            #         #new_stage = traci.simulation.findIntermodalRoute(current_edge, end_edge)[0] # Its a tuple with multiple stages
            #         #print(f"\nNew stage: {new_stage}\n")
            #         #traci.person.replaceStage(person_id, 0, new_stage)

            #         # # Another approach 
            #         # try:
            #         #     traci.person.rerouteTraveltime(person_id)
            #         # except traci.TraCIException as e:
            #         #     print(f"Error rerouting person {person_id}: {e}")


            #         # # Calculate remaining edges again
            #         # remaining_stages = traci.person.getRemainingStages(person_id)
            #         # remaining_edges = []
            #         # for stage in range(remaining_stages):
            #         #     remaining_edges.extend(traci.person.getEdges(person_id, stage))
                    
            #         print(f"\nPerson ID: {person_id}, Remaining edges after reroute: {remaining_edges}\n")

        disabled_crosswalks = []
        return disabled_crosswalks


    def _route_includes_disabled_crosswalk(self, remaining_edges, walking_lanes_to_disable):
        """
        Check if a route (list of edges) includes any of the disabled crosswalks.
        """

        to_lanes = []
        # For all the remaining edges of this pedestrian's route
        for i in range(len(remaining_edges) - 1):
            from_edge = remaining_edges[i]
            
            # Returns descriptions of the links outgoing from this lane [m]
            # A list containing id of successor lane together with priority
            connecting_lanes = traci.lane.getLinks(from_edge + "_0")  # Assume first lane of the edge
            #print(f"\nConnecting lanes: {connecting_lanes}\n")

            for to_lane, _, _, _, _, _, _, _ in connecting_lanes:
                to_lanes.append(to_lane)

        print(f"\nTo lanes: {to_lanes}\n")
        if any(to_lane in walking_lanes_to_disable for to_lane in to_lanes):
            return True

        return False
    
    ####################
            # Dont enable all
        # First: Enable all crosswalks (may have been disabled from previous action)
        for crosswalk_id in all_crosswalks:
            try:
                # Allow pedestrians on this crosswalk
                traci.lane.setAllowed(f"{crosswalk_id}_0", ["pedestrian"])

            except traci.exceptions.TraCIException as e:
                print(f"Error enabling crosswalk {crosswalk_id}: {e}")

        # Print the status of all crosswalks
        for crosswalk_id in all_crosswalks:
            try:
                allowed = traci.lane.getAllowed(f"{crosswalk_id}_0")
                #print(f"Crosswalk {crosswalk_id} is allowed: {allowed}")

            except traci.exceptions.TraCIException as e:
                print(f"Error getting allowed status for crosswalk {crosswalk_id}: {e}")

        # Failed approaches: Looking at the stages (A stage represents a segment of a person's overall trip)
        # For pedestrians, the total stages is always 1.
        # for person_id in traci.person.getIDList():
        #     total_stages = traci.person.getRemainingStages(person_id)
        #     print(f"\nPerson ID: {person_id}, Total stages: {total_stages}")
        #     # get edges for all stages
        #     for i in range(total_stages):
        #         # What is a stage?
        #         # What is current stage and next stage?
        #         stage_edges = traci.person.getEdges(person_id, i) #For walking stages this is the complete route. i is the value of next stage index ( value of 0 meaans current stage)
        #         print(f"Stage edge: {stage_edges}")
        #         for edge in stage_edges:
        #             if edge in incoming_walking_areas:
        #                 pedestrians_on_walking_areas += 1
        #             elif edge in connected_areas:
        #                 pedestrians_on_connected_areas += 1
        #             elif edge in incoming_edges:
        #                 pedestrians_on_incoming_edges += 1
        #             elif edge in outgoing_edges:
        #                 pedestrians_on_outgoing_edges += 1

        # Using traci.setDisallowed
        disabled_crosswalks = []
        pedestrian_ids = []
        for crosswalk_id in crosswalks_to_disable:
            try:
                pedestrians_on_incoming_edges = 0
                pedestrians_on_outgoing_edges = 0

                # Akk three, cosswalk themselves, the walking areas, and connected areas are edges.
                # Check if there are any pedestrians on the crosswalk
                pedestrians_on_crosswalk = len(traci.edge.getLastStepPersonIDs(crosswalk_id))
                pedestrian_ids.append(traci.edge.getLastStepPersonIDs(crosswalk_id)) # For debug, remove later

                # Get the incoming walking areas for this crosswalk
                incoming_walking_areas = self.controlled_crosswalks_dict[next(i for i, d in self.controlled_crosswalks_dict.items() if crosswalk_id in d['ids'])]['walking_areas']
                # print(f"\nIncoming walking areas: {incoming_walking_areas}")
                # Then, check for pedestrians on incoming walking areas
                pedestrians_on_walking_areas = sum(len(traci.edge.getLastStepPersonIDs(area)) for area in incoming_walking_areas)
                pedestrian_ids.append([traci.edge.getLastStepPersonIDs(area) for area in incoming_walking_areas]) # For debug, remove later

                # Check for pedestrians on connected_areas (could be a walking area or an edge)
                connected_areas = self.controlled_crosswalks_dict[next(i for i, d in self.controlled_crosswalks_dict.items() if crosswalk_id in d['ids'])]['connected_areas']
                pedestrians_on_connected_areas = sum(len(traci.edge.getLastStepPersonIDs(area)) for area in connected_areas)
                pedestrian_ids.append([traci.edge.getLastStepPersonIDs(area) for area in connected_areas]) # For debug, remove later

                print(f"Crosswalk {crosswalk_id}, pedestrians on:\n\tCrosswalks: {pedestrians_on_crosswalk}\n\tIncoming Walking areas: {pedestrians_on_walking_areas}\n\tConnected areas: {pedestrians_on_connected_areas}")

                # Get the related junctions and check in the incoming an outgoing edges.
                related_junctions = self.controlled_crosswalks_dict[next(i for i, d in self.controlled_crosswalks_dict.items() if crosswalk_id in d['ids'])]['related_junctions']
                
                for junction in related_junctions:
                    # Incoming edges from junction
                    incoming_edges = traci.junction.getIncomingEdges(junction)
                    for edge in incoming_edges:
                        pedestrians_on_edge = len(traci.edge.getLastStepPersonIDs(edge))
                        pedestrian_ids.append(traci.edge.getLastStepPersonIDs(edge)) # For debug, remove later
                        pedestrians_on_incoming_edges += pedestrians_on_edge
                    
                    # Outgoing edges from junction
                    outgoing_edges = traci.junction.getOutgoingEdges(junction)
                    for edge in outgoing_edges:
                        pedestrians_on_edge = len(traci.edge.getLastStepPersonIDs(edge))
                        pedestrian_ids.append(traci.edge.getLastStepPersonIDs(edge)) # For debug, remove later
                        pedestrians_on_outgoing_edges += pedestrians_on_edge

                    # This we dont have to check for each crossswalk. (1)
                    # For all pedestrians, 
                    for person_id in traci.person.getIDList():

                        # # Check the next edge
                        next_edge = traci.person.getNextEdge(person_id)
                        #print(f"Person ID: {person_id}, Next edge: {next_edge}")
                        
                        if next_edge in incoming_walking_areas:
                            pedestrians_on_walking_areas += 1

                        elif next_edge in connected_areas:
                            pedestrians_on_connected_areas += 1

                        elif next_edge in incoming_edges:
                            pedestrians_on_incoming_edges += 1

                        elif next_edge in outgoing_edges:
                            pedestrians_on_outgoing_edges += 1

                        # Check the next stages. Pedestrians only have one stage
                        stage_edges = traci.person.getEdges(person_id, 0) #For walking stages this is the complete route. i is the value of next stage index ( value of 0 meaans current stage)
                        # print(f"Pedestrian ID: {person_id}, Stage edges: {stage_edges}")
                        for edge in stage_edges:

                            if edge in incoming_walking_areas: # but here comparision is made with stuff of each crosswalk (2)
                                pedestrians_on_walking_areas += 1

                            elif edge in connected_areas:
                                pedestrians_on_connected_areas += 1

                            elif edge in incoming_edges:
                                pedestrians_on_incoming_edges += 1

                            elif edge in outgoing_edges:
                                pedestrians_on_outgoing_edges += 1

                print(f"\tPedestrians on incoming edges for crosswalk {crosswalk_id}: {pedestrians_on_incoming_edges}")
                print(f"\tPedestrians on outgoing edges for crosswalk {crosswalk_id}: {pedestrians_on_outgoing_edges}")

                if pedestrians_on_crosswalk == 0 and pedestrians_on_walking_areas == 0 and pedestrians_on_connected_areas == 0 \
                    and pedestrians_on_incoming_edges == 0 and pedestrians_on_outgoing_edges == 0:
                    # Disable pedestrians on this crosswalk
                    traci.lane.setDisallowed(f"{crosswalk_id}_0", ["pedestrian"])
                    disabled_crosswalks.append(crosswalk_id)
                    print(f"Disabled crosswalk {crosswalk_id}\n")

                else:
                    print(f"Skipped disabling crosswalk {crosswalk_id} due to presence of pedestrians\n")
                    #print(f"Pedestrian IDs: {pedestrian_ids}\n")

            except traci.exceptions.TraCIException as e:
                print(f"Error processing crosswalk {crosswalk_id}: {e}")

#####################

def _disallow_pedestrians(self, all_crosswalks, crosswalks_to_disable):
        """
        Operations before the sim or traci is started.
            Explicitly setting disallow does not work.
            Change the width (is good for appearance) and speed does not work either.
            Removing the crosswalk entirely also does not work.
        Operations after sim start/traci.

        # Cannot look into pedestrian's route too far in the future and re-route them. In the real world, we would not know their intentions like that.
        
        """

        # Based on the timesteps, allow a bunch and then disallow the bunch
        print(f"Step count: {self.step_count}") # Increments by 10 here

        # Define time ranges for disabling crosswalks
        time_ranges = [
            (200, 1000, all_crosswalks[:5]),
            (1000, 2000, all_crosswalks[5:]),
            (2000, float('inf'), all_crosswalks[:5])
        ]

        # Determine which crosswalks to disable based on current step
        crosswalks_to_disable = [] # TODO: remove this. For test
        for start, end, crosswalks in time_ranges:
            if start < self.step_count <= end:
                crosswalks_to_disable = crosswalks
                break
        
        #crosswalk_lanes_to_disable = [cw + "_0" for cw in crosswalks_to_disable]
        #print(f"\nCrosswalk lanes to disable: {crosswalk_lanes_to_disable}\n")
        
        print(f"\nCrosswalks to disable: {crosswalks_to_disable}\n")
        # The crosswalks do not directly fall into the route. However, the walking area associate with the corsswalk do.
        # Returns an empty list if the crosswalk is not in the dictionary
        walking_areas = []
        for crosswalk_id in crosswalks_to_disable:

            # First disable that crosswalk.

            walking_areas.extend(self.crosswalk_to_walking_areas.get(crosswalk_id, []))
        walking_lanes_to_disable = [wa + "_0" for wa in walking_areas]
        print(f"\nWalking lanes to disable: {walking_lanes_to_disable}\n")

        # Based on crosswalks to disable, first get the ids of all pedestrians who want to use this crosswalk.
        # Get all pedestrians in the simulation
        all_pedestrians = traci.person.getIDList()

        # Second find the pedestrians who are in the walking area of the crosswalk. Re-route them.
        # TODO: Right now, even if the pedestrian is far away from the crosswalk, they are rerouted.
        # Internal edge: The edges that begin with a : and are associated with junctions and interections.
        # When I use getRoadID, getNextEdge they return an internal edge. But when I get all edges using the stage information, I get the actual edge.
        for person_id in all_pedestrians:
            

            all_stages = traci.person.getRemainingStages(person_id)
            print(f"\nAll stages: {all_stages}\n")
            all_edges = [] 
            for stage in range(all_stages):
                all_edges.extend(traci.person.getEdges(person_id, stage))
            #all_edges = traci.person.getEdges(person_id, 0)  # 0 for the current (and only) stage
            print(f"\nAll edges: {all_edges}\n")

            current_edge = traci.person.getRoadID(person_id)
            print(f"\nCurrent edge: {current_edge}\n")

            position = traci.person.getPosition(person_id)
            print(f"\nPosition: {position}\n")
            
            current_index = all_edges.index(current_edge)
            #next_edge = traci.person.getNextEdge(person_id)
            #print(f"\nNext edge: {next_edge}\n")

            #next_index = 1# all_edges.index(next_edge)
            #current_edge = all_edges[next_index - 1]  # The edge before the next edge

            # print(f"(\nCurrent edge: {current_edge}\n") 
            # current_index = 0 #all_edges.index(current_edge)

            # # Find the index of the current edge in the full route
            # try:
            #     current_index = all_edges.index(current_edge)
            # except ValueError:
            #     # If current edge is not in the list (e.g., person is on an internal edge),
            #     # assume they're at the start of their route
            #     current_index = 0

            # Get only the remaining edges
            remaining_edges = all_edges[current_index:]
            #print(f"\nRemaining edges: {remaining_edges}\n")

            print(f"\n Person id: {person_id}")
            # Check if the pedestrian's remaining route includes a disabled crosswalk
            if self._route_includes_disabled_crosswalk(remaining_edges, walking_lanes_to_disable):
                print(f"\nPedestrian: {person_id} goes through the crosswalk\n")

                # Reroute the pedestrian to avoid the disabled crosswalk
                end_edge = remaining_edges[-1]
                
                print(f"\nCurrent edge: {current_edge}, End edge: {end_edge}\n")
                new_stage = traci.simulation.findIntermodalRoute(person_id, current_edge, end_edge)
                print(f"\nNew stage: {new_stage}\n")

        disabled_crosswalks = []
        return disabled_crosswalks


    def _route_includes_disabled_crosswalk(self, remaining_edges, walking_lanes_to_disable):
        """
        Check if a route (list of edges) includes any of the disabled crosswalks.
        """

        to_lanes = []
        # For all the remaining edges of this pedestrian's route
        for i in range(len(remaining_edges) - 1):
            from_edge = remaining_edges[i]
            
            # Returns descriptions of the links outgoing from this lane [m]
            # A list containing id of successor lane together with priority
            connecting_lanes = traci.lane.getLinks(from_edge + "_0")  # Assume first lane of the edge
            #print(f"\nConnecting lanes: {connecting_lanes}\n")

            for to_lane, _, _, _, _, _, _, _ in connecting_lanes:
                to_lanes.append(to_lane)

        print(f"\nTo lanes: {to_lanes}\n")
        if any(to_lane in walking_lanes_to_disable for to_lane in to_lanes):
            return True

        return False


self.direction_and_edges = {
            'upside': ['1050677005#10', '1050677005#11', '1050677005#12', '1050677005#13', '1050677005#14', '1050677005#15', 
                        '1050677005#16', '1050677005#18', '1050677005#19', '1050677005#2', '1050677005#20', '1050677005#21', 
                        '1050677005#3', '1050677005#4', '1050677005#5', '1050677005#6', '1050677005#7', '1050677005#8', '1050677005#9', 
                        '1050677007#1', '1051865729#2', '1051865729#3', '1054116926#4.273', '1054116928#1', '1054116929#0', '1054116929#5', 
                        '1054116929#6', '1054116929#7', '1054116929#9', '1054116932#0', '1054116932#1', '1054121747#2', '1054121752#0', 
                        '1054121752#1', '1054121753#1', '1058666207#1', '1060112789#2', '1060131391#1', '1060131401#3', '1060131408#1', 
                        '1060131409#1', '1060166262#1', '1060166262#2', '1098062416#0', '452522817#1', # Newly added below

                        ],

            'downside': ['1054116929#1', '1054116929#2', '1058666187#1', '1058666187#2', '1058666187#3', '1058666187#4', '1058666188#1', 
                        '1058666189#1', '1058666190#1', '1058666191#1', '1058666191#2', '1058666191#3', '1058666191#4', '1058666191#5', 
                        '1060112727#1', '1060112787#1', '1060112787#2', '1060112787#3', '1060112787#4', '1060112788', '1060112789#1', 
                        '1060112791#0', '1060131387#1', '1060131388#1', '1060131388#2', '1060131389#2', '1060131393#1', '1060131400#1', 
                        '1060131401#2', '1060131402', '1060131403#1', '1060131404#1', '1060131404#3', '1060131405#1', '1060131411#1', 
                        '1060166182#2', '1060166235#5.41', '1078803477#4.109', '1078803477#5', '1078803478#0', 'E8', '1060131390','1058666206#1', # Newly added below
                        
                        ]
            }