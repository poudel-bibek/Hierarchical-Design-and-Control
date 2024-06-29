import os
import json
import traci
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from sim_run import CraverRoadEnv
from models import MLPActorCritic

class PPO:
    """

    """
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip, ent_coef, vf_coef, device, batch_size):
        
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.batch_size = batch_size

        # Initialize the current policy network
        self.policy = MLPActorCritic(state_dim, action_dim, device).to(device)
        # Set up the optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize the old policy network (used for importance sampling)
        self.policy_old = MLPActorCritic(state_dim, action_dim, device).to(device)
        # Copy the parameters from the current policy to the old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def update(self, memory):
        """
        Update the policy and value networks using the collected experiences.
        
        TODO: Add support for GAE
        TODO: Use KL divergence instead of clipping

        """
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)): # Reverse the order
            if is_terminal: # Terminal timesteps have no future.
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward) # At the begining of the list, insert the calculated discounted reward
        
        # Convert rewards to tensor and z-score normalize them
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) # The 1e-5 prevents division by zero and prevents large values (stabilizes) 
        
        # Convert collected experiences to tensors
        old_states = torch.stack(memory.states).detach().to(self.device)
        old_actions = torch.stack(memory.actions).detach().to(self.device)
        old_logprobs = torch.stack(memory.logprobs).detach().to(self.device)
        
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
                policy_loss = -torch.min(surr1, surr2).mean() # Equation 7 in the paper
                value_loss = ((state_values - rewards) ** 2).mean()
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
    
class Memory:
    """
    Storage class for saving experience from interactions with the environment.
    """
    def __init__(self, device):
        self.device = device
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def append(self, state, action, logprob, reward, done):
        self.states.append(torch.FloatTensor(state).to(self.device))
        self.actions.append(torch.tensor(action).to(self.device))
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def save_config(args, model, save_path):
    """
    Save hyperparameters and model architecture to a JSON file.
    """
    config = {
        "hyperparameters": vars(args),
        "model_architecture": {
            "actor": str(model.policy.actor),
            "critic": str(model.policy.critic)
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)

def evaluate_traffic_light(args, env):
    """
    No algorithm is used here. For Benchmark
    For a given demand, evaluate the normal traffic light only scenario.
    We have to get somet things from the environment.
    """

    tl_ids = env.tl_ids
    phases = env.phases

    # Figure out the cycle lengths for each tl 
    cycle_lengths = {}
    for tl_id in tl_ids:
        phase = phases[tl_id]
        cycle_lengths[tl_id]  = sum([state['duration'] for state in phase])
       
    # Start the simulation
    # Things to add to call
    # "--no-step-log", "--no-warnings", "--time-to-teleport", "-1", "--waiting-time-memory", "10000", "--no-internal-links", "--no-duration-log", "--no-verbose"

    if args.auto_start:
        sumo_cmd = ["sumo-gui" if args.gui else "sumo", 
                    "--start" , 
                    "--quit-on-end", 
                    "-c", "./craver.sumocfg", 
                    '--step-length', str(args.step_length)]
                        
    else:
        sumo_cmd = ["sumo-gui" if args.gui else "sumo", 
                    "--quit-on-end", 
                    "-c", "./craver.sumocfg", 
                    '--step-length', str(args.step_length)]
    
    traci.start(sumo_cmd)

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


def main(args):
    
    env = CraverRoadEnv(args)
    
    if args.only_tl:

        if args.manual_demand_vehicles is None or args.manual_demand_pedestrians is None:
            print("Manual demand is None. Please specify a demand for both vehicles and pedestrians.")
            return

        evaluate_traffic_light(args, env)
        env.close()
    
    else: # PPO
        
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        print(f"Using device: {device}")

        print(f"\nDefined observation space: {env.observation_space}")
        print(f"Observation space shape: {env.observation_space.shape}")
        print(f"\nDefined action space: {env.action_space}")
        print(f"Action space shape: {env.action_space.n}\n")

        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        action_dim = env.action_space.n
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}\n")

        ppo = PPO(state_dim, action_dim, args.lr, args.gamma, args.K_epochs, args.eps_clip, args.ent_coef, args.vf_coef, device, args.batch_size)
        memory = Memory(device)

        # TensorBoard setup
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', current_time)
        os.makedirs('runs', exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # Save hyperparameters and model architecture
        config_path = os.path.join(log_dir, 'config.json')
        save_config(args, ppo, config_path)
        print(f"Configuration saved to {config_path}")

        # Model saving setup
        save_dir = os.path.join('saved_models', current_time)
        os.makedirs(save_dir, exist_ok=True)
        best_reward = float('-inf')

        total_episodes = args.total_timesteps // args.max_timesteps
        total_action_timesteps_per_episode = args.max_timesteps // args.action_duration # The total number of times actions will be taken in one episode.
        # There is an interal for loop for action_duration times.
        
        action_timesteps = 0 # Count how many action timesteps (not total) elapsed # Also measures how long this episode was (in-case of early termination or truncation)

        for ep in range(total_episodes):
            state, _ = env.reset()
            state = state.flatten()

            print(f"Initial observation (flattened): {state}")
            print(f"Initial observation (flattened) shape: {state.shape}\n")
            total_reward = 0 # To calculate the average reward per episode
            
            for t in range(total_action_timesteps_per_episode):
                print("-----------------------------------")
                print(f"\nStep: {action_timesteps}")
                action_timesteps += 1
                total_timesteps = ep * total_action_timesteps_per_episode + t
                
                state_tensor = torch.FloatTensor(state).to(device)
                action, log_prob = ppo.policy_old.act(state_tensor) # Policy old is used to act and collect experiences
                
                # These reward and next_state are for the action_duration timesteps.
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward

                # Saving experience in memory
                memory.append(state, action, log_prob, reward, done)
                
                #print(f"\nNext state: type: {type(next_state)}, shape:{next_state.shape}\n")
                state = next_state.flatten()

                # Update PPO every n times action has been taken
                if action_timesteps % args.update_freq == 0:
                    loss = ppo.update(memory)
                    memory.clear_memory()

                    print(f"Episode: {ep}/{total_episodes} (total timesteps: {total_timesteps}) \t Total Loss: {loss['total_loss']:.2f}")
                    # Loss is logged every time the model is updated.
                    if loss is not None: # TODO: Make this to check any
                        writer.add_scalars('Losses/Update', {
                        'Policy': loss['policy_loss'],
                        'Value': loss['value_loss'],
                        'Entropy': loss['entropy_loss'],
                        'Total': loss['total_loss']
                            }, total_timesteps)
                    else:
                        print("Warning: loss is None")

                if done or truncated: # Support for episode truncation based on crash or other unwanted events.
                    average_reward_per_episode = total_reward/action_timesteps

                    writer.add_scalars('Episode Metrics', {
                    'Average Reward': average_reward_per_episode, # Reward is logged at the end of each episode.
                    'Episode Length': action_timesteps
                        }, ep)
                    
                    # Logging
                    print(f'Episode: {ep}/{total_episodes} (total timesteps: {total_timesteps}) \t Average reward per episode: {average_reward_per_episode:.2f} ')
                    
                    # Save model periodically
                    if args.save_freq > 0 and ep % args.save_freq == 0:
                        torch.save(ppo.policy.state_dict(), os.path.join(save_dir, f'model_episode_{ep}.pth'))
                    
                    # Save best model
                    if average_reward_per_episode > best_reward:
                        best_reward = average_reward_per_episode
                        torch.save(ppo.policy.state_dict(), os.path.join(save_dir, 'best_average_reward_model.pth'))
                    
                    # Reset for next episode
                    state, _ = env.reset()
                    break

        env.close()
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation with PPO.')

    # Simulation
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI (default: False)')
    parser.add_argument('--step_length', type=float, default=1.0, help='Simulation step length (default: 1.0)') # What is one unit of increment in the simulation?
    parser.add_argument('--action_duration', type=float, default=10, help='Duration of each action (default: 10.0)') # How many simulation steps does each action occur for. 
    parser.add_argument('--auto_start', action='store_true', default=True, help='Automatically start the simulation')
    parser.add_argument('--vehicle_input_trips', type=str, default='./original_vehtrips.xml', help='Original Input trips file')
    parser.add_argument('--vehicle_output_trips', type=str, default='./scaled_vehtrips.xml', help='Output trips file')
    parser.add_argument('--pedestrian_input_trips', type=str, default='./original_pedtrips.xml', help='Original Input pedestrian trips file')
    parser.add_argument('--pedestrian_output_trips', type=str, default='./scaled_pedtrips.xml', help='Output pedestrian trips file')

    # If required to manually scale the demand (this happens automatically every episode as part of reset).
    parser.add_argument('--manual_demand_vehicles', type=float, default=None, help='Manually scale vehicle demand before starting the simulation')
    parser.add_argument('--manual_demand_pedestrians', type=float, default=None, help='Manually scale pedestrian demand before starting the simulation')
    parser.add_argument('--demand_scale_min', type=float, default=0.5, help='Minimum demand scaling factor for automatic scaling (default: 0.5)')
    parser.add_argument('--demand_scale_max', type=float, default=5.0, help='Maximum demand scaling factor for automatic scaling (default: 5.0)')

    # PPO
    #parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available (default: use CPU)')
    parser.add_argument('--total_timesteps', type=int, default=300000, help='Total number of timesteps the simulation will run (default: 300000)')
    parser.add_argument('--max_timesteps', type=int, default=1500, help='Maximum number of steps in one episode (default: 500)')
    parser.add_argument('--update_freq', type=int, default=128, help='Number of action timesteps between each policy update (default: 128)')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--K_epochs', type=int, default=4, help='Number of epochs to update policy (default: 4)')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter for PPO (default: 0.2)')
    parser.add_argument('--save_freq', type=int, default=10, help='Save model every n episodes (default: 10, 0 to disable)')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')

    # Evaluations
    parser.add_argument('--only_tl', action='store_true', help='Only use traffic lights for evaluation')

    args = parser.parse_args()
    main(args)