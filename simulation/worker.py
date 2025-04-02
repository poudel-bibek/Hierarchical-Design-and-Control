import os
import time
import random
import numpy as np
import torch
from ppo.ppo import PPO
from ppo.ppo_utils import Memory
from simulation.control_env import ControlEnv

def parallel_train_worker(rank, 
                         shared_policy_old, 
                         control_args, 
                         train_queue, 
                         worker_seed,
                         num_proposals,
                         lower_state_normalizer, 
                         lower_reward_normalizer,
                         higher_reward_normalizer,
                         extreme_edge_dict,
                         worker_device, 
                         network_iteration):
    """
    At every iteration, a number of workers will each parallelly carry out one episode in control environment.
    - Worker environment runs in CPU (SUMO runs in CPU).
    - Worker policy inference runs in GPU.
    - memory_queue is used to store the memory of each worker and send it back to the main process.
    - A shared policy_old (dict copy passed here) is used for importance sampling.
    - 1 memory transfer happens every memory_transfer_freq * action_duration sim steps.
    """

    # Set seed for this worker
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    worker_env = ControlEnv(control_args, worker_id=rank, network_iteration=network_iteration)
    memory_transfer_freq = control_args['memory_transfer_freq']  # Get from config
    local_memory = Memory() # A worker instance must have their own memory 

    state, _ = worker_env.reset(extreme_edge_dict, num_proposals)
    ep_reward = 0
    steps_since_update = 0

    for _ in range(control_args['total_action_timesteps_per_episode']):
        state = torch.FloatTensor(state)
        # Select action
        with torch.no_grad():
            state = lower_state_normalizer.normalize(state)
            state = state.to(worker_device)
            action, logprob = shared_policy_old.act(state, num_proposals) # sim runs in CPU, state will initially always be in CPU.
            value = shared_policy_old.critic(state.unsqueeze(0)) # add a batch dimension

            state = state.detach().cpu().numpy() # 2D
            action = action.detach().cpu().numpy() # 1D
            value = value.item() # Scalar
            logprob = logprob.item() # Scalar

        # Perform action
        # These reward and next_state are for the action_duration timesteps.
        next_state, control_reward, done, truncated, _ = worker_env.train_step(action) # need the returned state to be 2D
        control_reward = lower_reward_normalizer.normalize(control_reward).item()
        ep_reward += control_reward

        # Store data in memory
        local_memory.append(state, action, value, logprob, control_reward, done) 
        steps_since_update += 1

        if steps_since_update >= memory_transfer_freq or done or truncated:
            # Put local memory in the queue for the main process to collect
            train_queue.put((rank, local_memory, None))
            local_memory = Memory()  # Reset local memory
            steps_since_update = 0

        state = next_state
        if done or truncated:
            break
    
    # Higher level agent's reward can only be obtained after the lower level workers have finished
    design_reward_tensor = worker_env._get_design_reward(num_proposals).clone().detach().to(dtype=torch.float32, device='cpu')
    design_reward = higher_reward_normalizer.normalize(design_reward_tensor).item()
    print(f"Design reward: {design_reward}")

    # In PPO, we do not make use of the total reward. We only use the rewards collected in the memory.
    worker_env.close()
    time.sleep(5) # Essential
    del worker_env
    print(f"Worker {rank} finished. Control Episode Reward: {round(ep_reward, 2)}. Design Reward: {round(design_reward, 2)}. Worker puts None in queue.")
    train_queue.put((rank, None, design_reward))  # Signal that this worker is done 
    


def parallel_eval_worker(rank, 
                         eval_worker_config, 
                         eval_queue, 
                         tl=False, 
                         unsignalized=False):
    """
    - For the same demand, each worker runs n_iterations number of episodes and measures performance metrics at each iteration.
    - Each episode runs on a different random seed.
    - Performance metrics: 
        - Average waiting time (Veh, Ped)
        - Average travel time (Veh, Ped)
    - Returns a dictionary with performance metrics in all iterations.
    - For PPO: 
        - Create a single shared policy, and share among workers.
    - For TL:
        - Just pass tl = True
        - If unsignalized, all midblock TLs have no lights (equivalent to having all phases green)
    """
    
    worker_result = None # Initialize worker_result to None
    env = None # Initialize env to None
    worker_demand_scale = eval_worker_config.get('worker_demand_scale', 'unknown_scale') # Get scale early for finally block

    try:
        shared_policy = eval_worker_config['shared_policy']
        control_args = eval_worker_config['control_args']

        # We set the demand manually (so that automatic scaling does not happen)
        control_args['manual_demand_veh'] = worker_demand_scale
        control_args['manual_demand_ped'] = worker_demand_scale
        env = ControlEnv(control_args, worker_id=rank)
        worker_result = {} # Initialize results dict here
        
        # Run the worker
        for i in range(eval_worker_config['n_iterations']):
            worker_result[i] = {}

            SEED = random.randint(0, 1000000)
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

            worker_result[i]['SEED'] = SEED
            worker_device = eval_worker_config['worker_device']
            shared_eval_normalizer = eval_worker_config['shared_eval_normalizer']
            # Run the worker (reset includes warmup)
            state, _ = env.reset(tl = tl)
            veh_waiting_time_this_episode = 0
            ped_waiting_time_this_episode = 0
            veh_unique_ids_this_episode = 0
            ped_unique_ids_this_episode = 0

            with torch.no_grad():
                for _ in range(eval_worker_config['total_action_timesteps_per_episode']):
                    state = torch.FloatTensor(state)
                    state = shared_eval_normalizer.normalize(state)
                    state = state.to(worker_device)

                    action, _ = shared_policy.act(state, eval_worker_config['num_proposals'])
                    action = action.detach().cpu() # sim runs in CPU
                    state, reward, done, truncated, _ = env.eval_step(action, tl, unsignalized=unsignalized)

                    # During this step, get all vehicles and pedestrians
                    veh_waiting_time_this_step = env.get_vehicle_waiting_time()
                    ped_waiting_time_this_step = env.get_pedestrian_waiting_time()

                    veh_waiting_time_this_episode += veh_waiting_time_this_step
                    ped_waiting_time_this_episode += ped_waiting_time_this_step

                    veh_unique_ids_this_episode, ped_unique_ids_this_episode = env.total_unique_ids()
                    
                    if done or truncated:
                        break # Exit inner loop if episode ends early

            # gather performance metrics
            worker_result[i]['total_veh_waiting_time'] = veh_waiting_time_this_episode
            worker_result[i]['total_ped_waiting_time'] = ped_waiting_time_this_episode
            # Add safety check for division by zero
            worker_result[i]['veh_avg_waiting_time'] = (veh_waiting_time_this_episode / veh_unique_ids_this_episode) if veh_unique_ids_this_episode > 0 else 0
            worker_result[i]['ped_avg_waiting_time'] = (ped_waiting_time_this_episode / ped_unique_ids_this_episode) if ped_unique_ids_this_episode > 0 else 0
            worker_result[i]['total_conflicts'] = env.total_conflicts
            worker_result[i]['total_switches'] = env.total_switches

    except Exception as e:
        print(f"Error in parallel_eval_worker {rank} for demand {worker_demand_scale}: {e}")
        # worker_result remains None or whatever partial result was collected
        # Depending on needs, you could set worker_result = {'error': str(e)} 
        worker_result = {'error': str(e)} # Let's explicitly mark it as an error
        
    finally:
        # Ensure the environment is closed if it was created
        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"Error closing environment in worker {rank}: {e}")
            # time.sleep(10) # Consider if this sleep is truly essential or was for debugging
            del env
        
        # Always put a result (or error indicator) onto the queue
        try:
            eval_queue.put((worker_demand_scale, worker_result))
            print(f"Worker {rank} (Demand: {worker_demand_scale}) finished and put result on queue.")
        except Exception as e:
            print(f"Error putting result onto queue from worker {rank}: {e}")
            # If putting to queue fails, there's little else the worker can do.