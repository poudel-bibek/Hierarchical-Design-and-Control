def get_config():
    config = {
        # Simulation
        "sweep": False,  # Use wandb sweeps for hyperparameter tuning
        "gui": True,  # Use SUMO GUI (default: False)
        "step_length": 1.0,  # Simulation step length (default: 1.0). Since we have pedestrians, who walk slow. A value too small is not required.
        "action_duration": 10,  # Duration of each action (default: 10.0)
        "auto_start": True,  # Automatically start the simulation
        "vehicle_input_trips": "./SUMO_files/original_vehtrips.xml",  # Original Input trips file
        "vehicle_output_trips": "./SUMO_files/scaled_vehtrips.xml",  # Output trips file
        "pedestrian_input_trips": "./SUMO_files/original_pedtrips.xml",  # Original Input pedestrian trips file
        "pedestrian_output_trips": "./SUMO_files/scaled_pedtrips.xml",  # Output pedestrian trips file
        "original_net_file": "./SUMO_files/original_craver_road.net.xml",  # Original net file

        # Demand scaling
        "manual_demand_veh": None,  # Manually scale vehicle demand before starting the simulation (veh/hr)
        "manual_demand_ped": None,  # Manually scale pedestrian demand before starting the simulation (ped/hr)
        "demand_scale_min": 0.5,  # Minimum demand scaling factor for automatic scaling
        "demand_scale_max": 4.0,  # Maximum demand scaling factor for automatic scaling

        # PPO (general params)
        "seed": None,  # Random seed (default: None)
        "gpu": True,  # Use GPU if available (default: use CPU)
        "total_timesteps": 1500000,  # Total number of timesteps the simulation will run
        "max_timesteps": 720,  # Maximum number of steps in one episode
        "total_sweep_trials": 128,  # Total number of trials for the wandb sweep
        "memory_transfer_freq": 16,  # Frequency of memory transfer from worker to main process (Only applicable for lower level agent)

        # PPO (higher level agent)
        "higher_anneal_lr": True,  # Anneal learning rate
        "higher_lr": 0.001,  # Learning rate for higher-level agent
        "higher_gamma": 0.99,  # Discount factor for higher-level agent
        "higher_K_epochs": 4,  # Number of epochs to update policy for higher-level agent
        "higher_eps_clip": 0.2,  # Clip parameter for PPO for higher-level agent
        "higher_ent_coef": 0.01,  # Entropy coefficient for higher-level agent
        "higher_vf_coef": 0.5,  # Value function coefficient for higher-level agent
        "higher_batch_size": 32,  # Batch size for higher-level agent
        "higher_gae_lambda": 0.95,  # GAE lambda for higher-level agent
        "higher_hidden_channels": 64,  # Number of hidden channels in GATv2 layers
        "higher_out_channels": 32,  # Number of output channels in GATv2 layers
        "higher_initial_heads": 8,  # Number of attention heads in first GATv2 layer
        "higher_second_heads": 1,  # Number of attention heads in second GATv2 layer
        "higher_action_hidden_channels": 32,  # Number of hidden channels in action layers
        "higher_gmm_hidden_dim": 64,  # Hidden dimension for GMM layers
        "higher_num_mixtures": 3,  # Number of mixtures in GMM

        # Higher-level agent specific arguments
        "max_proposals": 10,  # Maximum number of crosswalk proposals
        "min_thickness": 0.1,  # Minimum thickness of crosswalks
        "max_thickness": 10.0,  # Maximum thickness of crosswalks
        "min_coordinate": 0.0,  # Minimum coordinate for crosswalk placement
        "max_coordinate": 1.0,  # Maximum coordinate for crosswalk placement

        # PPO (lower level agent)
        "lower_anneal_lr": True,  # Anneal learning rate
        "lower_gae_lambda": 0.95,  # GAE lambda
        "lower_update_freq": 128,  # Number of action timesteps between each policy update
        "lower_lr": 0.002,  # Learning rate
        "lower_gamma": 0.99,  # Discount factor
        "lower_K_epochs": 4,  # Number of epochs to update policy
        "lower_eps_clip": 0.2,  # Clip parameter for PPO
        "save_freq": 2,  # Save model after every n updates (0 to disable)
        "lower_ent_coef": 0.01,  # Entropy coefficient
        "lower_vf_coef": 0.5,  # Value function coefficient
        "lower_batch_size": 32,  # Batch size
        "lower_num_processes": 6,  # Number of parallel processes to use (Lower level agent has multiple workers)
        "lower_kernel_size": 3,  # Kernel size for CNN
        "lower_model_size": "medium",  # Model size for CNN: 'small' or 'medium'
        "lower_dropout_rate": 0.2,  # Dropout rate for CNN
        "lower_action_dim": 6,  # Number of action logits (not the same as number of actions. think)

        # Evaluation
        "evaluate": None,  # Evaluation mode: 'tl' (traffic light), 'ppo', or None
        "model_path": None,  # Path to the saved PPO model for evaluation
    }

    return config

"""
Best found hyperparameters from the sweep:
    
"""
