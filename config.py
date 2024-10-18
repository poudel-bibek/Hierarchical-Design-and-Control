import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation with PPO.')
    
    # Simulation
    parser.add_argument('--sweep', action='store_true', help='Use wandb sweeps for hyperparameter tuning')
    parser.add_argument('--gui', action='store_true', default=True, help='Use SUMO GUI (default: False)')
    parser.add_argument('--step_length', type=float, default=1.0, help='Simulation step length (default: 1.0)') # Since we have pedestrians, who walk slow. A value too small is not required.
    parser.add_argument('--action_duration', type=float, default=10, help='Duration of each action (default: 10.0)')
    parser.add_argument('--auto_start', action='store_true', default=True, help='Automatically start the simulation')
    parser.add_argument('--vehicle_input_trips', type=str, default='./SUMO_files/original_vehtrips.xml', help='Original Input trips file')
    parser.add_argument('--vehicle_output_trips', type=str, default='./SUMO_files/scaled_vehtrips.xml', help='Output trips file')
    parser.add_argument('--pedestrian_input_trips', type=str, default='./SUMO_files/original_pedtrips.xml', help='Original Input pedestrian trips file')
    parser.add_argument('--pedestrian_output_trips', type=str, default='./SUMO_files/scaled_pedtrips.xml', help='Output pedestrian trips file')
    
    # Demand scaling
    parser.add_argument('--manual_demand_veh', type=float, default=None, help='Manually scale vehicle demand before starting the simulation (veh/hr) ')
    parser.add_argument('--manual_demand_ped', type=float, default=None, help='Manually scale pedestrian demand before starting the simulation (ped/hr)')
    parser.add_argument('--demand_scale_min', type=float, default=0.5, help='Minimum demand scaling factor for automatic scaling (default: 0.5)')
    parser.add_argument('--demand_scale_max', type=float, default=4.0, help='Maximum demand scaling factor for automatic scaling (default: 5.0)')
    
    # PPO (general params)
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: None)')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available (default: use CPU)')
    parser.add_argument('--total_timesteps', type=int, default=1500000, help='Total number of timesteps the simulation will run (default: 1.5M)')
    parser.add_argument('--max_timesteps', type=int, default=720, help='Maximum number of steps in one episode (default: 720)')
    parser.add_argument('--total_sweep_trials', type=int, default=128, help='Total number of trials for the wandb sweep')
    parser.add_argument('--memory_transfer_freq', type=int, default=16,help='Frequency of memory transfer from worker to main process') # Only applicable for lower level agent.

    # PPO (higher level agent)
    parser.add_argument('--higher_anneal_lr', action='store_true', default=True, help='Anneal learning rate (default: False)')
    parser.add_argument('--higher_gae_lambda', type=float, default=0.95, help='GAE lambda (default: 0.95)')
    parser.add_argument('--higher_update_freq', type=int, default=128, help='Number of action timesteps between each policy update (default: 128)')
    parser.add_argument('--higher_lr', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--higher_gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--higher_K_epochs', type=int, default=4, help='Number of epochs to update policy (default: 4)')
    parser.add_argument('--higher_eps_clip', type=float, default=0.2, help='Clip parameter for PPO (default: 0.2)')
    parser.add_argument('--higher_ent_coef', type=float, default=0.01, help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--higher_vf_coef', type=float, default=0.5, help='Value function coefficient (default: 0.5)')
    parser.add_argument('--higher_batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--higher_num_processes', type=int, default=1, help='Number of parallel processes to use') # Higher level agent only controls one process.
    parser.add_argument('--higher_model_choice', choices=['cnn', 'mlp'], default='cnn', help='Model choice: cnn (default) or mlp')
    # Related to policy
    parser.add_argument('--higher_kernel_size', type=int, default=3, help='Kernel size for CNN (default: 3)')
    parser.add_argument('--higher_model_size', choices=['small', 'medium'], default='medium', help='Model size for CNN: small or medium (default)')
    parser.add_argument('--higher_dropout_rate', type=float, default=0.2, help='Dropout rate for CNN (default: 0.2)')

    # Higher-level agent specific arguments
    parser.add_argument('--max_proposals', type=int, default=10, help='Maximum number of crosswalk proposals')
    parser.add_argument('--min_thickness', type=float, default=0.1, help='Minimum thickness of crosswalks')
    parser.add_argument('--max_thickness', type=float, default=10.0, help='Maximum thickness of crosswalks')
    parser.add_argument('--min_coordinate', type=float, default=0.0, help='Minimum coordinate for crosswalk placement')
    parser.add_argument('--max_coordinate', type=float, default=1.0, help='Maximum coordinate for crosswalk placement')
    parser.add_argument('--higher_hidden_channels', type=int, default=64, help='Number of hidden channels in GATv2 layers')
    parser.add_argument('--higher_out_channels', type=int, default=32, help='Number of output channels in GATv2 layers')
    parser.add_argument('--higher_initial_heads', type=int, default=8, help='Number of attention heads in first GATv2 layer')
    parser.add_argument('--higher_second_heads', type=int, default=1, help='Number of attention heads in second GATv2 layer')
    parser.add_argument('--higher_action_hidden_channels', type=int, default=32, help='Number of hidden channels in action layers')
    parser.add_argument('--higher_gmm_hidden_dim', type=int, default=64, help='Hidden dimension for GMM layers')
    parser.add_argument('--higher_num_mixtures', type=int, default=3, help='Number of mixtures in GMM')

    # PPO (lower level agent)
    parser.add_argument('--lower_anneal_lr', action='store_true', default=True, help='Anneal learning rate (default: False)')
    parser.add_argument('--lower_gae_lambda', type=float, default=0.95, help='GAE lambda (default: 0.95)')
    parser.add_argument('--lower_update_freq', type=int, default=128, help='Number of action timesteps between each policy update (default: 128)')
    parser.add_argument('--lower_lr', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--lower_gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--lower_K_epochs', type=int, default=4, help='Number of epochs to update policy (default: 4)')
    parser.add_argument('--lower_eps_clip', type=float, default=0.2, help='Clip parameter for PPO (default: 0.2)')
    parser.add_argument('--save_freq', type=int, default=2, help='Save model after every n updates (default: 2, 0 to disable)')
    parser.add_argument('--lower_ent_coef', type=float, default=0.01, help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--lower_vf_coef', type=float, default=0.5, help='Value function coefficient (default: 0.5)')
    parser.add_argument('--lower_batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lower_num_processes', type=int, default=6, help='Number of parallel processes to use') # Lower level agent has multiple workers (processes).
    parser.add_argument('--lower_model_choice', choices=['cnn', 'mlp'], default='cnn', help='Model choice: cnn (default) or mlp')
    # Related to policy
    parser.add_argument('--lower_kernel_size', type=int, default=3, help='Kernel size for CNN (default: 3)')
    parser.add_argument('--lower_model_size', choices=['small', 'medium'], default='medium', help='Model size for CNN: small or medium (default)')
    parser.add_argument('--lower_dropout_rate', type=float, default=0.2, help='Dropout rate for CNN (default: 0.2)')

    # Evaluation
    parser.add_argument('--evaluate', choices=['tl', 'ppo'], help='Evaluation mode: traffic light (tl), PPO (ppo), or both')
    parser.add_argument('--model_path', type=str, help='Path to the saved PPO model for evaluation')

    return parser.parse_args()

"""
Best found hyperparameters from the sweep:
    
"""
