import os
import json
import wandb
import torch
import random
import numpy as np
from datetime import datetime
from ppo.ppo import PPO
from ppo.ppo_utils import Memory, WelfordNormalizer
from config import get_config, classify_and_return_args
import torch.multiprocessing as mp
from torch_geometric.data import Batch
from sweep import HyperParameterTuner
from torch.utils.tensorboard import SummaryWriter
from utils import *
from simulation.control_env import ControlEnv
from simulation.design_env import DesignEnv
from simulation.worker import parallel_eval_worker
from utils import load_policy, save_config

def train(train_config, is_sweep=False, sweep_config=None):
    """
    The lower level (control) agent is present inside the design environment as lower_ppo.
    lower level actor parallelization occurs inside step function of design environment. 
    Policy updates in both agents are serial/ centralized.
    """
    SEED = train_config['seed'] if train_config['seed'] else random.randint(0, 1000000)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda") if train_config['gpu'] and torch.cuda.is_available() else torch.device("cpu") 
    print(f"\nRandom seed: {SEED} \nUsing device: {device}")

    # Set and save hyperparameters 
    if is_sweep:
        for key, value in sweep_config.items():
            train_config[key] = value

    os.makedirs('runs', exist_ok=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(log_dir, f'config_{current_time}.json')
    save_config(train_config, config_path)
    print(f"\nConfiguration saved to {config_path}")

    design_args, control_args, higher_ppo_args, lower_ppo_args, eval_args = classify_and_return_args(train_config, device)

    # Print stats from dummy environment
    dummy_envs = {
        'higher' : DesignEnv(design_args, control_args, lower_ppo_args), 
        'lower' : ControlEnv(control_args, worker_id=None)
    }

    print(f"\nEnvironments")
    for env_type, env in dummy_envs.items():
        print(f"\n{env_type}-level:")
        print(f"\tObservation space: {env.observation_space}")
        print(f"\tObservation space shape: {env.observation_space.shape}")
        print(f"\tAction space: {str(env.action_space).replace('\n', '\n\t')}")
        if env_type == 'lower':
            print(f"\tAction dimension: {train_config['lower_action_dim']}")
        else: 
            print(f"\tIn channels: {train_config['higher_in_channels']}, Action dimension: {train_config['max_proposals']}\n")

    # use the dummy env shapes to init normalizers
    lower_obs_shape = dummy_envs['lower'].observation_space.shape
    eval_args['lower_state_dim'] = lower_obs_shape
    lower_state_normalizer = WelfordNormalizer(lower_obs_shape)
    lower_reward_normalizer = WelfordNormalizer(1)
    higher_reward_normalizer = WelfordNormalizer(1) # No state normalizer for design agent.

    dummy_envs['lower'].close() 
    dummy_envs['higher'].close()

    # Model saving and tensorboard 
    writer = SummaryWriter(log_dir=log_dir)
    save_dir = os.path.join('saved_models', current_time)
    os.makedirs(save_dir, exist_ok=True)
    eval_args['eval_save_dir'] = os.path.join('results', f'train_{current_time}')
    os.makedirs(eval_args['eval_save_dir'], exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    control_args.update({'writer': writer})
    control_args.update({'save_dir': save_dir})
    control_args.update({'global_seed': SEED})
    control_args.update({'total_action_timesteps_per_episode': train_config['lower_max_timesteps'] // train_config['lower_action_duration']})

    # Instead of using total_episodes, we will use total_iterations. 
    # Every iteration, num_process control agents interact with the environment for total_action_timesteps_per_episode steps (which further internally contains action_duration steps)
    total_iterations = train_config['total_timesteps'] // (train_config['lower_max_timesteps'] * train_config['lower_num_processes'])
    total_updates_higher = total_iterations // train_config['higher_update_freq']
    total_updates_lower = (train_config['total_timesteps'] / train_config['lower_action_duration']) // train_config['lower_update_freq']

    higher_ppo = PPO(**higher_ppo_args)
    higher_env = DesignEnv(design_args, control_args, lower_ppo_args, is_sweep=is_sweep, is_eval=False)
    higher_env.total_updates_lower = total_updates_lower

    higher_ppo.policy_old = higher_ppo.policy_old.to(device)
    higher_ppo.policy = higher_ppo.policy.to(device)

    higher_state = Batch.from_data_list([higher_env.reset()]) # Batch the data before sending to the model.
    print(f"\nHigher state at reset: {higher_state}")
    higher_memory = Memory()

    higher_update_count = 0
    current_lr_higher = higher_ppo_args['lr']
    
    for iteration in range(1, total_iterations + 1):
        print(f"\nStarting iteration: {iteration}/{total_iterations} with {higher_env.global_step} total steps so far\n")

        # Higher level agent takes node features, edge index, edge attributes and batch (to make single large graph) as input 
        # To produce padded fixed-sized actions num_actual_proposals is also returned.
        higher_ppo.policy_old.eval()
        padded_proposals, num_proposals, higher_logprob = higher_ppo.policy_old.act(higher_state, 
                                                                                        iteration, 
                                                                                        design_args['clamp_min'], 
                                                                                        design_args['clamp_max'], 
                                                                                        device,
                                                                                        visualize=True) 
        
        # Since the higher agent internally takes a step where a number of parallel lower agents take their own steps, 
        # We return things relevant to both the higher and lower agents. First, for higher.
        higher_next_state, higher_reward, higher_done, higher_info = higher_env.step(padded_proposals, 
                                                                                     num_proposals, 
                                                                                     iteration,
                                                                                     SEED,
                                                                                     lower_state_normalizer,
                                                                                     lower_reward_normalizer,
                                                                                     eval_args,
                                                                                     is_sweep)
        
        # Get value from critic network
        with torch.no_grad():
            critic_output = higher_ppo.policy_old.critic(higher_state)
            print(f"Critic output shape: {critic_output.shape}")
            print(f"Critic output: {critic_output}")
            # The critic returns a tensor of shape [1], so we can directly call .item()
            higher_value = critic_output.item()
        
        higher_memory.append(higher_state, padded_proposals, higher_value, higher_logprob, higher_reward, higher_done)

        if iteration % train_config['higher_update_freq'] == 0:
            higher_update_count += 1
            if train_config['higher_anneal_lr']:
                current_lr_higher = higher_ppo.update_learning_rate(higher_update_count, total_updates_higher)

            higher_ppo.update(higher_memory)
            # Reset memory
            del higher_memory
            higher_memory = Memory()

        higher_state = Batch.from_data_list([higher_next_state]) # Convert Data object back to Batch

        # Log higher level agent stuff.
        if is_sweep:
            wandb.log({ "higher_avg_reward": higher_reward,
                        "higher_current_lr": current_lr_higher if train_config['higher_anneal_lr'] else higher_ppo_args['lr'],
                        "global_step": higher_env.global_step          
                        })
        else:
            writer.add_scalar('Higher/Average_Reward', higher_reward, higher_env.global_step)
            writer.add_scalar('Higher/Current_LR', current_lr_higher if train_config['higher_anneal_lr'] else higher_ppo_args['lr'], higher_env.global_step)
    
    if is_sweep:
        wandb.finish()
    else:
        writer.close() # TODO: close writer for both agents?

def eval(control_args, 
         ppo_args, 
         eval_args, 
         policy_path=None, 
         tl=False, 
         unsignalized=False):
    """
    Works to evaluate a policy during training as well as stand-alone policy vs real-world TL (tl = True) evaluation.
    - Each demand is run on a different worker
    - Results saved as json dict. 
    """
    n_workers = eval_args['eval_n_workers']
    n_iterations = eval_args['eval_n_iterations']
    eval_device = torch.device("cuda") if eval_args['eval_worker_device']=='gpu' and torch.cuda.is_available() else torch.device("cpu")
    eval_demand_scales = eval_args['in_range_demand_scales'] + eval_args['out_of_range_demand_scales']
    all_results = {}

    eval_ppo = PPO(**ppo_args)
    shared_eval_normalizer = WelfordNormalizer(eval_args['state_dim'])
    shared_eval_normalizer.eval()
    if policy_path:
        load_policy(eval_ppo.policy, shared_eval_normalizer, policy_path)

    shared_policy = eval_ppo.policy.to(eval_device)
    shared_policy.share_memory()
    shared_policy.eval()
    
    # number of times the n_workers have to be repeated to cover all eval demands
    num_times_workers_recycle = len(eval_demand_scales) if len(eval_demand_scales) < n_workers else (len(eval_demand_scales) // n_workers) + 1
    for i in range(num_times_workers_recycle):
        start = n_workers * i   
        end = n_workers * (i + 1)
        demand_scales_evaluated_current_cycle = eval_demand_scales[start: end]

        eval_queue = mp.Queue()
        eval_processes = []  
        active_eval_workers = []
        demand_scale_to_rank = {}
        for rank, demand_scale in enumerate(demand_scales_evaluated_current_cycle): 
            demand_scale_to_rank[demand_scale] = rank
            print(f"For demand: {demand_scale}")    
            worker_config = {
                'n_iterations': n_iterations,
                'total_action_timesteps_per_episode': config['eval_n_timesteps'] // control_args['action_duration'], # Each time
                'worker_demand_scale': demand_scale,
                'shared_policy': shared_policy,
                'control_args': control_args,
                'worker_device': eval_device,
                'shared_eval_normalizer': shared_eval_normalizer,
                'num_proposals': control_args['max_proposals']
            }
            p = mp.Process(
                target=parallel_eval_worker,
                args=(rank, worker_config, eval_queue, tl, unsignalized))
            
            p.start()
            eval_processes.append(p)
            active_eval_workers.append(rank)

        while active_eval_workers:
            worker_demand_scale, result = eval_queue.get() #timeout=60) # Result is obtained after all iterations are complete
            print(f"Result from worker with demand scale: {worker_demand_scale}: {result}")
            all_results[worker_demand_scale] = result
            active_eval_workers.remove(demand_scale_to_rank[worker_demand_scale])

        for p in eval_processes:
            p.join()

    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    del eval_queue
    del shared_policy

    print(f"All results: {all_results}")    
    if tl and unsignalized:
        tl_state = "unsignalized"
    elif tl:
        tl_state = "tl"
    else:
        tl_state = "ppo"
    
    result_json_path = os.path.join(eval_args['eval_save_dir'], f'{policy_path.split("/")[-1].split(".")[0]}_{tl_state}.json') # f'eval_{policy_path.split("/")[2].split(".")[0]}_{tl_state}.json
    with open(result_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    f.close()
    return result_json_path
    
def main(config):
    """
    Cannot create a bunch of connections in main and then pass them around. 
    Because each new worker needs a separate pedestrian and vehicle trips file.
    """
    # Set the start method for multiprocessing. It does not create a process itself but sets the method for creating a process.
    # Spawn means create a new process. There is a fork method as well which will create a copy of the current process.
    mp.set_start_method('spawn') 
    if config['evaluate']: 
        device = torch.device("cuda") if config['eval_worker_device']=='gpu' and torch.cuda.is_available() else torch.device("cpu")
        control_args, ppo_args, eval_args = classify_and_return_args(config, device)
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        os.makedirs(f'./results', exist_ok=True)
        os.makedirs(f'./results/eval_{current_time}', exist_ok=True)
        eval_args['eval_save_dir'] = os.path.join('results', f'eval_{current_time}')

        dummy_env = ControlEnv(control_args, worker_id=None)
        eval_args['state_dim'] = dummy_env.observation_space.shape
        
        ppo_results_path = eval(control_args, ppo_args, eval_args, policy_path=config['eval_model_path'], tl= False)
        tl_results_path = eval(control_args, ppo_args, eval_args, policy_path=None, tl= True, unsignalized=False) 
        unsignalized_results_path = eval(control_args, ppo_args, eval_args, policy_path=None, tl= True, unsignalized=True)

        plot_main_results(unsignalized_results_path, 
                          tl_results_path,
                          ppo_results_path,
                          in_range_demand_scales = eval_args['in_range_demand_scales'])

    elif config['sweep']:
        tuner = HyperParameterTuner(config, train)
        tuner.start()
    else:
        train(config)

if __name__ == "__main__":
    config = get_config()
    main(config)