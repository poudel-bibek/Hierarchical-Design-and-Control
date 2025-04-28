import wandb

class HyperParameterTuner: 
    def __init__(self, config, train_function):
        self.config = config
        self.project = "urban-design-and-control"
        self.train_function = train_function
        
    def start(self, ):
        sweep_id = self.create_sweep_config()
        wandb.agent(sweep_id, function= self.hyperparameter_tune_main, count= self.config["total_sweep_trials"]) 

    def hyperparameter_tune_main(self):
        try:
            wandb.init(project=self.project, settings=wandb.Settings(disable_git=True))
            config = wandb.config
            self.train_function(self.config, is_sweep=True, sweep_config=config)
        finally:
            wandb.finish() 

    def create_sweep_config(self, method='random'): # options: random, grid, bayes
        """
        If using random, max and min values are required.
        We do not want to get weird weights such as 0.192 for various params. Hence not using random search.
        However, if using grid search requires all parameters to be categorical, constant, int_uniform

        On using bayes method for hyperparameter tuning:
            - Works well for small number of continuous parameters. Scales poorly.

        # What to maximize?
        Keep in mind: 
            1. Every iteration, the policy gets updated. 
            2. Each episode runs in a parallel worker with a randomly sampled scaling factor (ped/ veh demands).
            3. An episode might not be over yet the policy might be updated. This is how PPO works.
        Best Choice: avg_reward i.e., Average reward per process in this iteration.
            1. Robustness: avg_reward considers the performance across all processes in an iteration, each with potentially different demand scaling factors. 
            2. Consistency: By averaging rewards across processes, we reduce the impact of potential overfitting to a specific demand scaling factor.
        """

        sweep_config = {

            'method': method, 
            'metric': {
                'name': 'evals/avg_ped_arrival', # Using avg_eval like in bayes
                'goal': 'minimize'  # Minimize average evaluation time
                },

            'parameters': {
                'higher_lr': { 'values': [1e-4, 5e-4] },
                'lower_lr': { 'values': [5e-4, 1e-3] },
                # HRL Interaction / Update Frequencies
                'higher_update_freq': { 'values': [8, 16] },
                'lower_update_freq': { 'values': [1024, 2048] },
                # 'num_mixtures': { 'values': [5, 7, 10] },
                'higher_readout_k': { 'values': [32, 64] },
                # --- Higher-Level Specific ---
                'higher_batch_size': {'values': [2, 4]},
                'higher_eps_clip': {'values': [0.1, 0.2, 0.3]},
                'higher_ent_coef': {'values': [0.001, 0.005]},
                # --- Lower-Level Specific ---
                'lower_batch_size': {'values': [32, 64]},
                'lower_eps_clip': {'values': [0.1, 0.2, 0.3]},
                'lower_ent_coef': {'values': [0.005, 0.01, 0.02]},
                'lower_K_epochs': {'values': [2, 4, 8]},
            },
        }
        
        return wandb.sweep(sweep_config, entity="fluidic-city", project=self.project)

# learnings from sweep: 
# - Higher lr high
# - Higher entropy coeff low 