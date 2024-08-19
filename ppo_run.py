import os
import json
import wandb
import traci
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
import torch.multiprocessing as mp # wow we get this from torch itself

# from collections import deque
# from functools import partial
#import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from sim_run import CraverRoadEnv
from models import MLPActorCritic

class Memory:
    """
    Storage class for saving experience from interactions with the environment.
    These memories will be made in CPU but loaded in GPU for the policy update.
    """
    def __init__(self,):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def append(self, state, action, logprob, reward, done):
        self.states.append(torch.FloatTensor(state))
        self.actions.append(torch.tensor(action.clone().detach())) # Not sure why the clone is necessary (the action is sampled from the worker device = cpu). But got warning without.
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPO:
    """
    This implementation is parallelized using Multiprocessing i.e. multiple CPU cores each running a separate process.
    Multiprocessing vs Multithreading:
    - In the CPython implementation, the Global Interpreter Lock (GIL) is a mechanism used to prevent multiple threads from executing Python bytecodes at once. 
    - This lock is necessary because CPython is not thread-safe, i.e., if multiple threads were allowed to execute Python code simultaneously, they could potentially interfere with each other, leading to data corruption or crashes. 
    - The GIL prevents this by ensuring that only one thread can execute Python code at any given time.
    - Since only one thread can execute Python code at a time, programs that rely heavily on threading for parallel execution may not see the expected performance gains.
    - In contrast, multiprocessing allows multiple processes to execute Python code in parallel, bypassing the GIL and taking full advantage of multiple CPU cores.
    - However, multiprocessing has higher overhead than multithreading due to the need to create separate processes and manage inter-process communication.
    - In Multiprocessing, we create separate processes, each with its own Python interpreter and memory space
    """
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip, ent_coef, vf_coef, device, batch_size, num_processes):
        
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        self.num_processes = num_processes

        # Initialize the current policy network
        self.policy = MLPActorCritic(state_dim, action_dim, device).to(device)

        # Initialize the old policy network (used for importance sampling)
        self.policy_old = MLPActorCritic(state_dim, action_dim, device).to(device)

        # Copy the parameters from the current policy to the old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy.share_memory() # Share the policy network across all processes. Any tensor can be shared across processes by calling this.
        self.policy_old.share_memory() # Share the old policy network across all processes. 

        # Set up the optimizer for the current policy network
        self.initial_lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.initial_lr)
        self.total_iterations = None  # Will be set in the train function
    
    def update_learning_rate(self, iteration):
        """
        Linear annealing. At the end of training, the learning rate is 0.
        """
        if self.total_iterations is None:
            raise ValueError("total_iterations must be set before calling update_learning_rate")
        
        frac = 1.0 - (iteration / self.total_iterations)
        new_lr = frac * self.initial_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    
    def update(self, memories):
        """
        memories = combined memories from all processes.
        Update the policy and value networks using the collected experiences.
        
        TODO: Add support for GAE
        TODO: Use KL divergence instead of clipping

        """
        combined_memory = Memory()
        for memory in memories:
            combined_memory.actions.extend(memory.actions)
            combined_memory.states.extend(memory.states)
            combined_memory.logprobs.extend(memory.logprobs)
            combined_memory.rewards.extend(memory.rewards)
            combined_memory.is_terminals.extend(memory.is_terminals)

        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(combined_memory.rewards), reversed(combined_memory.is_terminals)): # Reverse the order
            if is_terminal: # Terminal timesteps have no future.
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward) # At the begining of the list, insert the calculated discounted reward
        
        # Convert rewards to tensor and z-score normalize them
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) # The 1e-5 prevents division by zero and prevents large values (stabilizes) 
        
        # Convert collected experiences to tensors
        old_states = torch.stack(combined_memory.states).detach().to(self.device)
        old_actions = torch.stack(combined_memory.actions).detach().to(self.device)
        old_logprobs = torch.stack(combined_memory.logprobs).detach().to(self.device)
        
        # Create a dataloader first, not everyone does this way.
        dataset = torch.utils.data.TensorDataset(old_states, old_actions, old_logprobs, rewards)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_loss = 0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for states, actions, old_logprobs_batch, rewards_batch in dataloader:

                # Evaluating old actions and values using current policy network
                logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
                
                # Finding the ratio (pi_theta / pi_theta_old) for imporatnce sampling (we want to use the samples obtained from old policy to get the new policy)
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())
                
                # Action Advantage = difference between expected return of taking the action and expected return of following the policy
                # First term is monte carlo estimate of the reward with discounting
                advantages = rewards_batch - state_values.detach() 

                # Finding Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                # Calculate policy and value losses
                # TODO: Is the mean necessary here? In policy loss and entropy loss.
                policy_loss = -torch.min(surr1, surr2).mean() # Equation 7 in the paper
                value_loss = ((state_values - rewards) ** 2).mean() # MSE 
                entropy_loss = dist_entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss # Equation 9 in the paper
                
                # Accumulate losses
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy_loss += entropy_loss.item()

                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        num_batches = len(dataloader) * self.K_epochs
        avg_policy_loss /= num_batches
        avg_value_loss /= num_batches
        avg_entropy_loss /= num_batches


        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Return the average batch loss per epoch
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_policy_loss + self.vf_coef * avg_value_loss - self.ent_coef * avg_entropy_loss
        }
    
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

            state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_dim = env.action_space.n
            ppo_model = MLPActorCritic(state_dim, action_dim, device).to(device)
            ppo_model.load_state_dict(torch.load(args.model_path, map_location=device)) 

            state, _ = env.reset()
            for t in range(args.max_timesteps):
                state_tensor = torch.FloatTensor(state.flatten()).to(device)
                action, _ = ppo_model.act(state_tensor)
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

    How frequently should a parallel worker send a memory to the main process?
    Lets set this to 8.
    """

    # Set seed for this worker
    worker_seed = global_seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    env = CraverRoadEnv(args)
    worker_device = torch.device("cpu")
    memory_transfer_freq = 8

    # The central memory is a collection of memories from all processes.
    # A worker instance must have their own memory 
    local_memory = Memory()
    shared_policy_old = shared_policy_old.to(worker_device)

    state, _ = env.reset()
    state = state.flatten()

    print(f"Worker {rank} started.")
    print(f"Initial observation (flattened): {state}")
    print(f"Initial observation (flattened) shape: {state.shape}\n")
    ep_reward = 0
    steps_since_update = 0
     
    for _ in range(args.total_action_timesteps_per_episode):
        state_tensor = torch.FloatTensor(state).to(worker_device)

        # Select action
        with torch.no_grad():
            action, logprob = shared_policy_old.act(state_tensor)

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

    print(f"Worker {rank} finished. Total reward: {ep_reward}")
    env.close()
    memory_queue.put((rank, None))  # Signal that this worker is done

def define_sweep_config():
    """
    If using random, max and min values are required.
    However, if using grid search requires all parameters to be categorical, constant, int_uniform
    """
    sweep_config = {
        'method': 'random', # options: random, grid, bayes
        'metric': {
            'name': 'reward',
            'goal': 'maximize'
        },

        # For random
        'parameters': {
            'lr': {'min': 1e-4, 'max': 1e-2},
            'gamma': {'min': 0.9, 'max': 0.999},
            'K_epochs': {'values': [2, 4, 8]},
            'eps_clip': {'min': 0.1, 'max': 0.3},
            'ent_coef': {'min': 0.001, 'max': 0.1},
            'vf_coef': {'min': 0.1, 'max': 1.0},
            'batch_size': {'values': [16, 32, 64]},
        }

        #  For grid
        # 'parameters': {
        #     'lr': {'values': [0.001, 0.002, 0.005] },
        #     'gamma': {'values': [0.95, 0.99, 0.999]},
        #     'K_epochs': {'values': [4, 8, 16] },
        #     'eps_clip': {'values': [0.1, 0.2, 0.3]},
        #     'ent_coef': {'values': [0.01, 0.05, 0.1]},
        #     'vf_coef': {'values': [0.5, 0.75, 1.0]},
        #     'batch_size': {'values': [32, 64, 128]},
        #     'num_processes': {'values': [4, 8, 16]}
        # }

    }

    return sweep_config

def train_sweep(config=None):
    """
    
    """
    with wandb.init(config=config):
        config = wandb.config
        
        # Update args with wandb config
        args.lr = config.lr
        args.gamma = config.gamma
        args.K_epochs = config.K_epochs
        args.eps_clip = config.eps_clip
        args.ent_coef = config.ent_coef
        args.vf_coef = config.vf_coef
        args.batch_size = config.batch_size
        
        # Call the main training function
        train(args, is_sweep=True)

def run_sweep():
    sweep_config = define_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="ppo_urban_and_traffic_control")
    wandb.agent(sweep_id, function=train_sweep, count=5) # 5 is the Number of sweeps trials to try 

def train(args, is_sweep=False):
    """
    Actors are parallelized i.e., create their own instance of the envinronment and interact with it (perform policy rollout).
    All aspects of training are centralized.
    Auto tune hyperparameters using wandb sweeps.

    Although Adam maintains an independent and changing lr for each policy parameter, there are still potential benefits of having a lr schedule
    Annealing is a special form of scheduling where the learning rate may not strictly decrease 

    """
    SEED = args.seed if args.seed else random.randint(0, 1000000)
    print(f"Random seed: {SEED}")

    # Set global seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    global_step = 0
    env = CraverRoadEnv(args) # First environment instance. Required for setup.
 
    # Spawn creates a new process. There is a fork method as well which will create a copy of the current process.
    mp.set_start_method('spawn') 

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")
    print(f"\nDefined observation space: {env.observation_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"\nDefined action space: {env.action_space}")
    print(f"Options per action dimension: {env.action_space.nvec}")

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = len(env.action_space.nvec)
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}\n")
    env.close() # We actually dont make use of this environment for any other stuff. Each worker will have their own environment.

    ppo = PPO(state_dim, 
            action_dim, 
            args.lr, 
            args.gamma, 
            args.K_epochs, 
            args.eps_clip, 
            args.ent_coef, 
            args.vf_coef, 
            device, 
            args.batch_size,
            args.num_processes)

    if is_sweep:
        # Wandb setup
        wandb.init(project="ppo_urban_and_traffic_control", config=args)
    else: 
        # TensorBoard setup
        # No need to save the model during sweep.
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', current_time)
        os.makedirs('runs', exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # Save hyperparameters and model architecture
        config_path = os.path.join(log_dir, f'config_{current_time}.json')
        save_config(args, SEED, ppo, config_path)
        print(f"Configuration saved to {config_path}")

        # Model saving setup
        save_dir = os.path.join('saved_models', current_time)
        os.makedirs(save_dir, exist_ok=True)
        best_reward = float('-inf')

    # In a parallel setup, instead of using total_episodes, we will use total_iterations. 
    # In each iteration, multiple actors interact with the environment for max_timesteps. i.e., each iteration will have num_processes episodes.
    total_iterations = args.total_timesteps // (args.max_timesteps * args.num_processes)
    ppo.total_iterations = total_iterations # For lr annealing
    args.total_action_timesteps_per_episode = args.max_timesteps // args.action_duration # Each actor will run for max_timesteps and each timestep will have action_duration steps.
    
    # Counter to keep track of how many times action has been taken 
    action_timesteps = 0
    memory_queue = mp.Queue()

    for iteration in range(total_iterations):
        print(f"\nIteration: {iteration}/{total_iterations}", end = "\t")

        processes = [] # Create a list of processes
        #manager = mp.Manager() # Facilitates communication and sharing of data between processes

        for rank in range(args.num_processes):
            p = mp.Process(target=worker, args=(rank, args, ppo.policy_old, memory_queue, SEED)) # Create a process to execute the worker function
            p.start()
            processes.append(p)

        # Essential for synchronization. However, we are not performing updates to the policy only after the episode is over.
        # We are updating the policy every n times action has been taken. (See PPO paper). Hence this is disabled.
        # The join() method is called on each process in the processes list. This ensures that the main program waits for all processes to complete before continuing.
        # for p in processes:
        #     p.join()

        if args.anneal_lr:
            current_lr = ppo.update_learning_rate(iteration)

        all_memories = []
        active_workers = set(range(args.num_processes))

        while active_workers:
            rank, memory = memory_queue.get()

            if memory is None:
                active_workers.remove(rank)

            else:
                all_memories.append(memory)
                print(f"Memory from worker {rank} received. Memory size: {len(memory.states)}")

                # Look at the size of the memory and update action_timesteps
                action_timesteps += len(memory.states)

                # Update PPO every n times action has been taken
                if action_timesteps % args.update_freq == 0:
                    loss = ppo.update(all_memories)
        
                    # clear memory to prevent memory growth 
                    for memory in all_memories:
                        memory.clear_memory()

                    total_reward = sum(sum(memory.rewards) for memory in all_memories)
                    avg_reward = total_reward / args.num_processes # Average reward per process in this iteration
                    print(f", Average Reward: {avg_reward:.2f}")
                    
                    # reset all memories
                    all_memories = []

                    # Logging every time the model is updated.
                    if loss is not None:

                        if is_sweep: # Wandb for hyperparameter tuning
                            
                            wandb.log({     "iteration": iteration,
                                            "avg_reward": avg_reward,
                                            "policy_loss": loss['policy_loss'],
                                            "value_loss": loss['value_loss'],
                                            "entropy_loss": loss['entropy_loss'],
                                            "total_loss": loss['total_loss'],
                                            "current_lr": current_lr,
                                            "global_step": global_step          })
                        
                        else: # Tensorboard for regular training
                            global_step = iteration * args.num_processes + action_timesteps*args.action_duration
                            total_updates = int(action_timesteps / args.update_freq)

                            writer.add_scalar('Rewards/Average_Reward', avg_reward, global_step)
                            writer.add_scalar('Updates/Total_Policy_Updates', total_updates, global_step)
                            writer.add_scalar('Losses/Policy_Loss', loss['policy_loss'], global_step)
                            writer.add_scalar('Losses/Value_Loss', loss['value_loss'], global_step)
                            writer.add_scalar('Losses/Entropy_Loss', loss['entropy_loss'], global_step)
                            writer.add_scalar('Losses/Total_Loss', loss['total_loss'], global_step)
                            writer.add_scalar('Learning_Rate/Current_LR', current_lr, global_step)
                            print(f"Logged data at step {global_step}")

                            # Save model every n times it has been updated (Important: Not every iteration)
                            if args.save_freq > 0 and total_updates % args.save_freq == 0:
                                torch.save(ppo.policy.state_dict(), os.path.join(save_dir, f'model_iteration_{iteration+1}.pth'))

                            # Save best model so far
                            if avg_reward > best_reward:
                                best_reward = avg_reward
                                torch.save(ppo.policy.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                                
                    else: # For some reason..
                        print("Warning: loss is None")

    if is_sweep:
        wandb.finish()
    else:
        writer.close()

def main(args):
    """
    Keep main short.
    """

    if args.evaluate:  # Eval TL or PPO
        if args.manual_demand_veh is None or args.manual_demand_ped is None:
            print("Manual demand is None. Please specify a demand for both vehicles and pedestrians.")
            return None
        
        else: 
            env = CraverRoadEnv(args)
            run_data, all_directions = evaluate_controller(args, env)
            calculate_performance(run_data, all_directions, args.step_length)
            env.close()

    elif args.sweep:
        run_sweep()

    else:
        train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation with PPO.')
    parser.add_argument('--sweep', action='store_true', help='Use wandb sweeps for hyperparameter tuning')
    # Simulation
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI (default: False)')
    parser.add_argument('--step_length', type=float, default=1.0, help='Simulation step length (default: 1.0)') # What is one unit of increment in the simulation?
    parser.add_argument('--action_duration', type=float, default=10, help='Duration of each action (default: 10.0)') # How many simulation steps does each action occur for. 
    parser.add_argument('--auto_start', action='store_true', default=True, help='Automatically start the simulation')
    parser.add_argument('--vehicle_input_trips', type=str, default='./SUMO_files/original_vehtrips.xml', help='Original Input trips file')
    parser.add_argument('--vehicle_output_trips', type=str, default='./SUMO_files/scaled_vehtrips.xml', help='Output trips file')
    parser.add_argument('--pedestrian_input_trips', type=str, default='./SUMO_files/original_pedtrips.xml', help='Original Input pedestrian trips file')
    parser.add_argument('--pedestrian_output_trips', type=str, default='./SUMO_files/scaled_pedtrips.xml', help='Output pedestrian trips file')

    # If required to manually scale the demand (this happens automatically every episode as part of reset).
    parser.add_argument('--manual_demand_veh', type=float, default=None, help='Manually scale vehicle demand before starting the simulation')
    parser.add_argument('--manual_demand_ped', type=float, default=None, help='Manually scale pedestrian demand before starting the simulation')
    parser.add_argument('--demand_scale_min', type=float, default=1.0, help='Minimum demand scaling factor for automatic scaling (default: 0.5)')
    parser.add_argument('--demand_scale_max', type=float, default=2.0, help='Maximum demand scaling factor for automatic scaling (default: 5.0)')

    # PPO
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: None)')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available (default: use CPU)')
    parser.add_argument('--total_timesteps', type=int, default=500000, help='Total number of timesteps the simulation will run (default: 300000)')
    parser.add_argument('--max_timesteps', type=int, default=1500, help='Maximum number of steps in one episode (default: 500)')
    parser.add_argument('--anneal_lr', action='store_true', help='Anneal learning rate (default: False)')

    # The default update freq in the PPO paper is 128 but in our case, the interval between actions itself is 10 timesteps.
    parser.add_argument('--update_freq', type=int, default=128, help='Number of action timesteps between each policy update (default: 128)')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--K_epochs', type=int, default=4, help='Number of epochs to update policy (default: 4)')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter for PPO (default: 0.2)')
    parser.add_argument('--save_freq', type=int, default=2, help='Save model after every n updates (default: 2, 0 to disable)')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--num_processes', type=int, default=10, help='Number of parallel processes to use')
    
    # Evaluations
    parser.add_argument('--evaluate', choices=['tl', 'ppo'], help='Evaluation mode: traffic light (tl), PPO (ppo), or both')
    parser.add_argument('--model_path', type=str, help='Path to the saved PPO model for evaluation')
    args = parser.parse_args()
    main(args)