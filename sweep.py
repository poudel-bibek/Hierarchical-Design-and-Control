import wandb
import torch.nn as nn
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

    def create_sweep_config(self, method='bayes'): # options: random, grid, bayes
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
                'name': 'avg_eval', # Using avg_eval like in bayes
                'goal': 'minimize'  # Minimize average evaluation time
                },

            'parameters': {
                # Higher Level Agent (Design)
                'higher_lr': {
                    'values': [5e-5, 1e-4, 5e-4, 1e-3]
                },
                'higher_gae_lambda': {
                    'values': [0.95, 0.98, 0.99]
                },
                'higher_update_freq': {
                    'values': [2] #[8, 16, 32]
                },
                'higher_gamma': {
                    'values': [0.98, 0.99, 0.995]
                },
                'higher_K_epochs': {
                    'values': [2, 4, 8]
                },
                'higher_eps_clip': {
                    'values': [0.15, 0.2, 0.25]
                },
                'higher_ent_coef': {
                    'values': [0.005, 0.01, 0.02]
                },
                'higher_vf_coef': {
                    'values': [0.4, 0.5, 0.6]
                },
                'higher_vf_clip_param': {
                    'values': [0.4, 0.5, 0.6]
                },
                'higher_batch_size': {
                    'values': [16, 32, 64]
                },
                'higher_dropout_rate': {
                    'values': [0.1, 0.2, 0.3]
                },
                'higher_model_size': {
                    'values': ['medium'] 
                },
                'higher_activation': {
                    'values': ["tanh", "relu"]
                },
                'initial_heads': { 
                    'values': [4, 8]
                },
                'higher_hidden_channels': { 
                    'values': [32, 64, 128]
                },
                'higher_out_channels': { 
                    'values': [16, 32, 64]
                },

                # Lower Level Agent (Control)
                # 'lower_lr': {
                #     'values': [5e-5, 1e-4, 2e-4]
                # },
                # 'lower_gae_lambda': {
                #     'values': [0.95, 0.98, 0.99]
                # },
                # 'lower_update_freq': {
                #     'values': [512, 1024, 2048]
                # },
                # 'lower_gamma': {
                #     'values': [0.98, 0.99, 0.995]
                # },
                # 'lower_K_epochs': {
                #     'values': [2, 4, 8]
                # },
                # 'lower_eps_clip': {
                #     'values': [0.15, 0.2, 0.25]
                # },
                # 'lower_ent_coef': {
                #     'values': [0.005, 0.01, 0.02]
                # },
                # 'lower_vf_coef': {
                #     'values': [0.4, 0.5, 0.6]
                # },
                # 'lower_vf_clip_param': {
                #     'values': [0.4, 0.5, 0.6]
                # },
                # 'lower_batch_size': {
                #     'values': [32, 64, 128]
                # },
                # 'lower_dropout_rate': {
                #     'values': [0.1, 0.2, 0.3]
                # },
                # 'lower_model_size': {
                #     'values': ['medium']
                # },
                # 'lower_activation': {
                #     'values': ["tanh", "relu"]
                # },
            }
        }
        
        return wandb.sweep(sweep_config, entity="fluidic-city", project=self.project)
