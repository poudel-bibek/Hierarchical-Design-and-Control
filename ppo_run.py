import os
import json
import wandb
wandb.require("core") # Bunch of improvements in using the core.

import traci
import queue
import torch
import random
import numpy as np
import torch.multiprocessing as mp # wow we get this from torch itself

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from craver_control_env import CraverControlEnv
from craver_design_env import CraverDesignEnv
from models import MLPActorCritic, CNNActorCritic

from wandb_sweep import HyperParameterTuner
from ppo_alg import PPO, Memory
from config import get_args

def save_config(args, SEED, model, save_path):
    """
    Save hyperparameters and model architecture to a JSON file.
    """
    config = {
        "hyperparameters": vars(args),
        "global_seed": SEED,
        "model_architecture": {
            "actor": str(model.policy.actor),
            "critic": str(model.policy.critic)
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)

def evaluate_controller(args, env):
    """
    For benchmarking.
    Evaluate either the traffic light or PPO as the controller.
    TODO: Make the evaluation N number of times each with different seeds. Report average results.
    """

    # Collect step data separated at each landmarks such as TL lights
    step_data = []
    if args.evaluate == 'tl':
        tl_ids = env.tl_ids
        phases = env.phases

        # Figure out the cycle lengths for each tl 
        cycle_lengths = {}
        for tl_id in tl_ids:
            phase = phases[tl_id]
            cycle_lengths[tl_id]  = sum([state['duration'] for state in phase])
        
        if args.auto_start:
            sumo_cmd = ["sumo-gui" if args.gui else "sumo", 
                        "--start" , 
                        "--quit-on-end", 
                        "-c", "./SUMO_files/craver.sumocfg", 
                        '--step-length', str(args.step_length)]
                            
        else:
            sumo_cmd = ["sumo-gui" if args.gui else "sumo", 
                        "--quit-on-end", 
                        "-c", "./SUMO_files/craver.sumocfg", 
                        '--step-length', str(args.step_length)]
        
        traci.start(sumo_cmd)
        env.sumo_running = True
        env._initialize_lanes()

        # Now run the sim till the horizon
        for t in range(args.max_timesteps):
            for tl_id in tl_ids:

                # using t, determine where in the cycle we are
                current_pos_in_cycle = t % cycle_lengths[tl_id]

                # Find the index/ state
                state_index = 0
                for state in phases[tl_id]:
                    current_pos_in_cycle -= state['duration']
                    if current_pos_in_cycle < 0:
                        break
                    state_index += 1

                # Set the state
                state_string = phases[tl_id][state_index]['state']
                traci.trafficlight.setRedYellowGreenState(tl_id, state_string)

            # This is outside the loop
            traci.simulationStep()

            # After the simulation step 
            occupancy_map = env._get_occupancy_map()
            corrected_occupancy_map = env._step_operations(occupancy_map, print_map=True, cutoff_distance=100)
            step_info, all_directions = collect_step_data(t, corrected_occupancy_map, env)

            print(f"Step: {t}, step info: {step_info}")
            step_data.append(step_info)

    elif args.evaluate == 'ppo':
        if args.model_path:
            # device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
            # Maybe we should use only CPU during evaluation
            device = torch.device("cpu")
            model_choice_functions = {
                'cnn': CNNActorCritic,
                'mlp': MLPActorCritic,
            }
            
            if args.model_choice == 'mlp':
                model_kwargs = {
                    'hidden_dim': 256, 
                }
            else: 
                action_dim = env.action_space.n
                n_channels = 1 
                model_kwargs = {
                    'action_duration': env.observation_space.shape[0],  
                    'per_timestep_state_dim': env.observation_space.shape[1], 
                    'model_size': args.model_size,  
                    'kernel_size': args.kernel_size,
                    'dropout_rate': args.dropout_rate
                }

            ppo_model = model_choice_functions[args.model_choice](n_channels, action_dim, device, **model_kwargs).to(device) 
            ppo_model.load_state_dict(torch.load(args.model_path, map_location=device)) 

            state, _ = env.reset()
            for t in range(args.max_timesteps):
                action, _ = ppo_model.act(state)
                state, reward, done, truncated, info = env.step(action)
                
                occupancy_map = env._get_occupancy_map()
                corrected_occupancy_map = env._step_operations(occupancy_map, print_map=False, cutoff_distance=100)

                step_info, all_directions = collect_step_data(t, corrected_occupancy_map, env)
                step_data.append(step_info)

                if done or truncated:
                    break

        else:
            print("Model path not provided. Cannot evaluate PPO.")
            return None

    else: 
        print("Invalid evaluation mode. Please choose either 'tl' or 'ppo'.")
        return None
    
    return step_data, all_directions

def collect_step_data(step, occupancy_map, env):
    """
    Collect detailed data for a single step using the occupancy map.
    A vehicle is considered to be waiting (in a queue) if the velocity is less than 0.5 m/s.

    Avreage waiting time: On average, how long does a vehicle wait while crossing the intersection? 
    """

    step_info = {'step': step}
    all_directions = [f"{direction}-{turn}" for direction in env.directions for turn in env.turns]
    
    for tl_id, tl_data in occupancy_map.items():
        step_info[tl_id] = {
            'vehicle': {
                'queue_length': {direction: 0 for direction in all_directions},
                'total_outgoing': [] , # Total vehicles that crossed the intersection. Per step. From one step to the next, there might be repitition. Needs to be filtered later.
            },
            'pedestrian': {}
        }

    # Collect vehicle IDs and data for each traffic light
    for tl_id, tl_data in occupancy_map.items():

        # For queue, process both incoming and inside vehicles
        for movement_direction in tl_data['vehicle'].keys():  

            if movement_direction in ['incoming', 'inside']:
                for lane_group, ids in tl_data['vehicle'][movement_direction].items():

                    for veh_id in ids:
                        veh_velocity = traci.vehicle.getSpeed(veh_id)

                        # Increment queue length if vehicle is waiting
                        if veh_velocity < 1.0:
                            # Ensure the lane_group exists in our queue_length dictionary
                            if lane_group in step_info[tl_id]['vehicle']['queue_length']:

                                step_info[tl_id]['vehicle']['queue_length'][lane_group] += 1
                            else:
                                print(f"Warning: Unexpected lane group '{lane_group}' encountered.")

            # For total outgoing vehicles
            else: 
                for _, vehicles in tl_data['vehicle']['outgoing'].items():
                    step_info[tl_id]['vehicle']['total_outgoing'].extend(vehicles)

    return step_info, all_directions

def calculate_performance(run_data, all_directions, step_length):
    """
    Calculate the performance metrics from the run data.
    1. Average Waiting Time: For every outgoing vehicle, on average what is the waiting time?
    2. Average Queue Length: For every direction (12 total), on average what is the queue length? Counted whenever there is a queue.
    3. Overall Average Queue Length: Average queue length across all directions.
    4. Throughput: Number of vehicles per hour that crossed the intersection.
    """

    total_waiting_time = 0
    unique_outgoing_vehicles = set()
    queue_lengths = {direction: [] for direction in all_directions}
    
    # Process each step's data
    for step_info in run_data:
        for tl_id, tl_data in step_info.items():
            if tl_id != 'step':  # Skip the 'step' key
                # Collect unique outgoing vehicle IDs
                unique_outgoing_vehicles.update(tl_data['vehicle']['total_outgoing'])
                
                # Sum queue lengths
                for direction, length in tl_data['vehicle']['queue_length'].items():
                    if length > 0:
                        queue_lengths[direction].append(length)
                        total_waiting_time += length # Each waiting vehicle contributes 1 timestep of waiting time

    total_outgoing_vehicles = len(unique_outgoing_vehicles)
    
    # Calculate average waiting time using the actual step length
    total_simulation_waiting_time = total_waiting_time * step_length # In seconds
    avg_waiting_time = total_simulation_waiting_time / total_outgoing_vehicles 
    
    avg_queue_lengths = {direction: sum(lengths) / len(lengths) if lengths else 0
                         for direction, lengths in queue_lengths.items()}
    
    all_queue_lengths = [length for lengths in queue_lengths.values() for length in lengths]
    overall_avg_queue_length = sum(all_queue_lengths) / len(all_queue_lengths) if all_queue_lengths else 0

    total_simulation_time = (run_data[-1]['step'] * step_length)/ 3600  # Convert to hours
    throughput = total_outgoing_vehicles / total_simulation_time # Vehicles per hour
    
    # Print results
    print("\nPerformance Metrics:")
    print(f"Total Unique Outgoing Vehicles: {total_outgoing_vehicles}")
    print(f"Average Waiting Time: {avg_waiting_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} vehicles/hour")
    print(f"Overall Average Queue Length: {overall_avg_queue_length:.2f}")
    print("\nAverage Queue Lengths by Direction:")
    for direction, avg_length in avg_queue_lengths.items():
        print(f"  {direction}: {avg_length:.2f}")

def worker(rank, args, shared_policy_old, memory_queue, global_seed):
    """
    The device for each worker is always CPU.
    At every iteration, 1 worker will carry out one episode.
    memory_queue is used to store the memory of each worker and send it back to the main process.
    shared_policy_old is used for importance sampling.

    Although the worker runs simulation in CPU, the policy inference is done in GPU.
    """

    # Set seed for this worker
    worker_seed = global_seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    env = CraverControlEnv(args, worker_id=rank)
    worker_device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    memory_transfer_freq = args.memory_transfer_freq  # Get from args

    # The central memory is a collection of memories from all processes.
    # A worker instance must have their own memory 
    local_memory = Memory()
    shared_policy_old = shared_policy_old.to(worker_device)

    state, _ = env.reset()
    ep_reward = 0
    steps_since_update = 0
     
    for _ in range(args.total_action_timesteps_per_episode):
        state_tensor = torch.FloatTensor(state).to(worker_device)

        # Select action
        with torch.no_grad():
            action, logprob = shared_policy_old.act(state_tensor)
            action = action.cpu()  # Explicitly Move to CPU, Incase they were on GPU
            logprob = logprob.cpu() 

        # Perform action
        # These reward and next_state are for the action_duration timesteps.
        next_state, reward, done, truncated, info = env.step(action)
        ep_reward += reward

        # Store data in memory
        local_memory.append(state, action, logprob, reward, done)
        steps_since_update += 1

        if steps_since_update >= memory_transfer_freq or done or truncated:
            # Put local memory in the queue for the main process to collect
            memory_queue.put((rank, local_memory))
            local_memory = Memory()  # Reset local memory
            steps_since_update = 0

        if done or truncated:
            break

        state = next_state.flatten()

    # In PPO, we do not make use of the total reward. We only use the rewards collected in the memory.
    print(f"Worker {rank} finished. Total reward: {ep_reward}")
    env.close()
    memory_queue.put((rank, None))  # Signal that this worker is done

def train(train_args, is_sweep=False, config=None):
    """
    Actors are parallelized i.e., create their own instance of the envinronment and interact with it (perform policy rollout).
    All aspects of training are centralized.
    Auto tune hyperparameters using wandb sweeps.

    Although Adam maintains an independent and changing lr for each policy parameter, there are still potential benefits of having a lr schedule
    Annealing is a special form of scheduling where the learning rate may not strictly decrease 

    TODO: For evaluation of a sweep, currently we are looking at reward (and not the traffic related metrics). 
    In the future, evals can be added here. i.e., evaluate the policy and then calculate the waiting time, queue length, throughput etc. metrics

    # Move towards two stage learning setup.
    # 1. Higher-level agent makes the design decisions.
    # 2. Lower-level agent makes the traffic control decisions.
    """

    SEED = train_args.seed if train_args.seed else random.randint(0, 1000000)
    print(f"Random seed: {SEED}")

    # Set global seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if is_sweep: #DO NOT MOVE THIS BELOW. # Update args with wandb config
        for key, value in config.items():
            if hasattr(train_args, key):
                setattr(train_args, key, value)
        
    global_step = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() and train_args.gpu else "cpu")
    print(f"Using device: {device}")

    # Dummy agents. Required for setup.
    agents = {'lower' : CraverControlEnv(train_args, worker_id=None) , 
              'higher':  CraverDesignEnv(train_args) }
    
    for agent in agents.keys():
        print(f"\nFor {agent} level agent:")
        print(f"\tDefined observation space: {agents[agent].observation_space}")
        print(f"\tObservation space shape: {agents[agent].observation_space.shape}")
        print(f"\tDefined action space: {agents[agent].action_space}")
        print(f"\tOptions per action dimension: {agents[agent].action_space.nvec}")

    # Higher level agent
    if train_args.higher_model_choice == 'mlp':
        state_dim_flat = 40 # TODO: harcoded, change this
        model_kwargs_higher = {
            'hidden_dim': 256,  # For MLP
            }
    else: # cnn
        state_dim = (10, 74) # TODO: hardcoded, change this
        n_channels = 1
        model_kwargs_higher = {
            'action_duration': train_args.action_duration,  
            'per_timestep_state_dim': 40,  # TODO: hardcoded, change this
            'model_size': train_args.higher_model_size,  
            'kernel_size': train_args.higher_kernel_size,
            'dropout_rate': train_args.higher_dropout_rate
            }
    
    higher_action_dim = 15 # TODO: hardcoded, change this
    print(f"\nHigher level agent: \n\tState dimension: {state_dim}, Action dimension: {higher_action_dim}\n")

    agents['higher'].close() # Dont need this anymore
    higher_ppo = PPO(state_dim_flat if train_args.higher_model_choice == 'mlp' else n_channels, 
        higher_action_dim, 
        device, 
        train_args.higher_lr, 
        train_args.higher_gamma, 
        train_args.higher_K_epochs, 
        train_args.higher_eps_clip, 
        train_args.higher_ent_coef, 
        train_args.higher_vf_coef, 
        train_args.higher_batch_size,
        train_args.higher_num_processes, # This agent has a single process
        train_args.higher_gae_lambda,
        train_args.higher_model_choice,
        agent_type="higher",
        **model_kwargs_higher
        )
    
    # Lower level agent
    # If model choice is mlp, the input is flat. However, if model choice is cnn, the input is single channel 2d
    if train_args.lower_model_choice == 'mlp':
        state_dim_flat = agents['lower'].observation_space.shape[0] * agents['lower'].observation_space.shape[1]
        model_kwargs_lower = {
            'hidden_dim': 256,  # For MLP
            }
    else: # cnn
        state_dim = agents['lower'].observation_space.shape # e.g., (10, 74) = (action_duration, per_timestep_state_dim)
        n_channels = 1
        model_kwargs_lower = {
            'action_duration': train_args.action_duration,  
            'per_timestep_state_dim': agents['lower'].observation_space.shape[1],  
            'model_size': train_args.lower_model_size,  
            'kernel_size': train_args.lower_kernel_size,
            'dropout_rate': train_args.lower_dropout_rate
            }
    lower_action_dim = len(agents['lower'].action_space.nvec)
    print(f"\nLower level agent: \n\tState dimension: {state_dim}, Action dimension: {lower_action_dim}")

    agents['lower'].close() # Dont need this anymore
    lower_ppo = PPO(state_dim_flat if train_args.lower_model_choice == 'mlp' else n_channels, 
        lower_action_dim, 
        device, 
        train_args.lower_lr, 
        train_args.lower_gamma, 
        train_args.lower_K_epochs, 
        train_args.lower_eps_clip, 
        train_args.lower_ent_coef, 
        train_args.lower_vf_coef, 
        train_args.lower_batch_size,
        train_args.lower_num_processes, # This agent has parallel processes (workers)
        train_args.lower_gae_lambda,
        train_args.lower_model_choice,
        agent_type="lower",
        **model_kwargs_lower
        )

    if not is_sweep: 
        # TensorBoard setup
        # No need to save the model during sweep.
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', current_time)
        os.makedirs('runs', exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # Save hyperparameters and model architecture
        config_path = os.path.join(log_dir, f'config_{current_time}.json')
        save_config(train_args, SEED, lower_ppo, config_path)
        print(f"Configuration saved to {config_path}")

        # Model saving setup
        save_dir = os.path.join('saved_models', current_time)
        os.makedirs(save_dir, exist_ok=True)
        best_reward = float('-inf')

    # Instead of using total_episodes, we will use total_iterations. 
    # In each iteration, multiple lower level agent actors interact with the environment for max_timesteps. i.e., each iteration will have num_processes episodes.
    # Each iteration is equivalent to a single timestep for the higher agent.
    total_iterations = train_args.total_timesteps // (train_args.max_timesteps * train_args.lower_num_processes)
    lower_ppo.total_iterations = total_iterations # For lr annealing
    train_args.total_action_timesteps_per_episode = train_args.max_timesteps // train_args.action_duration # Each actor will run for max_timesteps and each timestep will have action_duration steps.
    
    # Counter to keep track of how many times action has been taken 
    action_timesteps = 0
    for iteration in range(total_iterations):

        global_step = iteration * train_args.lower_num_processes + action_timesteps*train_args.action_duration
        print(f"\nStarting iteration: {iteration + 1}/{total_iterations} with {global_step} total steps so far\n")

        # Create a manager to handle shared objects
        manager = mp.Manager()
        memory_queue = manager.Queue()

        processes = []
        for rank in range(train_args.lower_num_processes):
            p = mp.Process(target=worker, args=(rank, train_args, lower_ppo.policy_old, memory_queue, SEED)) # Create a process to execute the worker function
            p.start()
            processes.append(p)

        if train_args.lower_anneal_lr:
            current_lr = lower_ppo.update_learning_rate(iteration)

        all_memories = []
        active_workers = set(range(train_args.lower_num_processes))

        while active_workers:
            try:
                rank, memory = memory_queue.get(timeout=60)  # Add a timeout to prevent infinite waiting
                
                if memory is None:
                    active_workers.remove(rank)
                else:
                    all_memories.append(memory)
                    print(f"Memory from worker {rank} received. Memory size: {len(memory.states)}")

                    # Look at the size of the memory and update action_timesteps
                    action_timesteps += len(memory.states)

                    # Update lower level PPO every n times action has been taken
                    if action_timesteps % train_args.lower_update_freq == 0:
                        loss = lower_ppo.update(all_memories)

                        total_reward = sum(sum(memory.rewards) for memory in all_memories)
                        avg_reward = total_reward / train_args.lower_num_processes # Average reward per process in this iteration
                        print(f"\nAverage Reward per process: {avg_reward:.2f}\n")
                        
                        # clear memory to prevent memory growth (after the reward calculation)
                        for memory in all_memories:
                            memory.clear_memory()

                        # reset all memories
                        del all_memories #https://pytorch.org/docs/stable/multiprocessing.html
                        all_memories = []

                        # Logging every time the model is updated.
                        if loss is not None:

                            if is_sweep: # Wandb for hyperparameter tuning
                                wandb.log({     "iteration": iteration,
                                                "avg_reward": avg_reward, # Set as maximize in the sweep config
                                                "policy_loss": loss['policy_loss'],
                                                "value_loss": loss['value_loss'],
                                                "entropy_loss": loss['entropy_loss'],
                                                "total_loss": loss['total_loss'],
                                                "current_lr_lower": current_lr if train_args.lower_anneal_lr else train_args.lr,
                                                "global_step": global_step          })
                            
                            else: # Tensorboard for regular training
                                
                                total_updates = int(action_timesteps / train_args.lower_update_freq)
                                writer.add_scalar('Rewards/Average_Reward', avg_reward, global_step)
                                writer.add_scalar('Updates/Total_Policy_Updates', total_updates, global_step)
                                writer.add_scalar('Losses/Policy_Loss', loss['policy_loss'], global_step)
                                writer.add_scalar('Losses/Value_Loss', loss['value_loss'], global_step)
                                writer.add_scalar('Losses/Entropy_Loss', loss['entropy_loss'], global_step)
                                writer.add_scalar('Losses/Total_Loss', loss['total_loss'], global_step)
                                writer.add_scalar('Learning_Rate/Current_LR', current_lr, global_step)
                                print(f"Logged data at step {global_step}")

                                # Save model every n times it has been updated (Important: Not every iteration)
                                if train_args.save_freq > 0 and total_updates % train_args.save_freq == 0:
                                    torch.save(lower_ppo.policy.state_dict(), os.path.join(save_dir, f'model_iteration_{iteration+1}.pth'))

                                # Save best model so far
                                if avg_reward > best_reward:
                                    best_reward = avg_reward
                                    torch.save(lower_ppo.policy.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                                    
                        else: # For some reason..
                            print("Warning: loss is None")

            except queue.Empty:
                print("Timeout waiting for worker. Continuing...")

        # At the end of an iteration, wait for all processes to finish
        # The join() method is called on each process in the processes list. This ensures that the main program waits for all processes to complete before continuing.
        for p in processes:
            p.join()

    if is_sweep:
        wandb.finish()
    else:
        writer.close()

def main(args):
    """
    Keep main short.
    We cannot create a bunch of connections in main and then pass them around. Because each new worker needs a separate pedestrian and vehicle trips file.
    """

    # Set the start method for multiprocessing. It does not create a process itself but sets the method for creating a process.
    # Spawn means create a new process. There is a fork method as well which will create a copy of the current process.
    mp.set_start_method('spawn', force=True) 
    mp.set_sharing_strategy('file_system')

    if args.evaluate:  # Eval TL or PPO
        if args.manual_demand_veh is None or args.manual_demand_ped is None:
            print("Manual demand is None. Please specify a demand for both vehicles and pedestrians.")
            return None
        
        else: 
            env = CraverControlEnv(args)
            run_data, all_directions = evaluate_controller(args, env)
            calculate_performance(run_data, all_directions, args.step_length)
            env.close()

    elif args.sweep:
        tuner = HyperParameterTuner(args)
        tuner.start()

    else:
        train(args)


if __name__ == "__main__":
    args = get_args()
    main(args)