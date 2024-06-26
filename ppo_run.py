import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from sim_run import CraverRoadEnv

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        """
        Define the neural networks for both actor and critic
        """
        super(ActorCritic, self).__init__()
        self.device = device
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim)
        ).to(device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).to(device)
        
    def act(self, state):
        """
        Select an action based on the current state
        """
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, states, actions):
        """
        Evaluate the actions given the states
        """
        action_probs = self.actor(states)
        dist = Categorical(logits=action_probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        
        return action_logprobs, state_values, dist_entropy
    
class PPO:
    """

    """
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip, ent_coef, vf_coef, device):
        
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        
        # Initialize the current policy network
        self.policy = ActorCritic(state_dim, action_dim, device).to(device)
        # Set up the optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize the old policy network (used for importance sampling)
        self.policy_old = ActorCritic(state_dim, action_dim, device).to(device)
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
        
        total_loss = 0.0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values using current policy network
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta_old) for imporatnce sampling (we want to use the samples obtained from old policy to get the new policy)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Action Advantage = difference between expected return of taking the action and expected return of following the policy
            # First term is monte carlo estimate of the reward with discounting
            advantages = rewards - state_values.detach() 

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Calculate policy and value losses
            policy_loss = -torch.min(surr1, surr2).mean() # Equation 7 in the paper
            value_loss = ((state_values - rewards) ** 2).mean()
            entropy_loss = dist_entropy.mean()
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss # Equation 9 in the paper
            total_loss += loss.item()

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Return the average loss per epoch
        return total_loss / self.K_epochs
    
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

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    env = CraverRoadEnv(args)

    print(f"\nDefined observation space: {env.observation_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"\nDefined action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.n}\n")

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}\n")

    ppo = PPO(state_dim, action_dim, args.lr, args.gamma, args.K_epochs, args.eps_clip, args.ent_coef, args.vf_coef, device)
    memory = Memory(device)

    # TensorBoard setup
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time)
    os.makedirs('runs', exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Model saving setup
    save_dir = os.path.join('saved_models', current_time)
    os.makedirs(save_dir, exist_ok=True)
    best_reward = float('-inf')

    state, _ = env.reset()
    state = state.flatten()
    print(f"Initial observation (flattened): {state}")
    print(f"Initial observation (flattened) shape: {state.shape}\n")

    total_timesteps = 0
    episode_reward = 0
    episode_length = 0
    episode_count = 0

    while total_timesteps < args.total_timesteps:
        for t in range(args.max_timesteps):
            total_timesteps += 1
            episode_length += 1
            
            state_tensor = torch.FloatTensor(state).to(device)
            action, log_prob = ppo.policy_old.act(state_tensor) # Policy old is used to act and collect experiences
            
            next_state, reward, done, truncated, info = env.step(action)
            
            # This cannot be how its done, initially even before the agent takes action. There will be a phase group.
            #env.current_phase_group = action

            # Saving experience in memory
            memory.append(state, action, log_prob, reward, done)
            
            #print(f"\nNext state: type: {type(next_state)}, shape:{next_state.shape}\n")
            state = next_state.flatten()
            episode_reward += reward

            # Update PPO every n timesteps
            if total_timesteps % args.update_freq == 0:
                loss = ppo.update(memory)
                memory.clear_memory()
                if loss is not None:
                    writer.add_scalar('Loss/Update', loss, total_timesteps // args.update_freq)
                else:
                    print("Warning: loss is None")

            if done or truncated: # Support for episode truncation based on crash or other unwanted events.
                episode_count += 1
                # TensorBoard logging
                writer.add_scalar('Reward/Episode', episode_reward, episode_count)
                writer.add_scalar('Episode Length', episode_length, episode_count)
                
                # Logging
                print(f'Episode {episode_count} \t Length: {episode_length} \t Reward: {episode_reward:.2f} \t Total Timesteps: {total_timesteps}')
                
                # Save model periodically
                if args.save_freq > 0 and episode_count % args.save_freq == 0:
                    torch.save(ppo.policy.state_dict(), os.path.join(save_dir, f'model_episode_{episode_count}.pth'))
                
                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(ppo.policy.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                
                # Reset for next episode
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                break

    env.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation with PPO.')

    # Simulation
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI (default: False)')
    parser.add_argument('--step_length', type=float, default=1.0, help='Simulation step length (default: 1.0)') # What is one unit of increment in the simulation?
    parser.add_argument('--auto_start', action='store_true', default=True, help='Automatically start the simulation')
    parser.add_argument('--vehicle_input_trips', type=str, default='./original_vehtrips.xml', help='Original Input trips file')
    parser.add_argument('--vehicle_output_trips', type=str, default='./scaled_vehtrips.xml', help='Output trips file')
    parser.add_argument('--pedestrian_input_trips', type=str, default='./original_pedtrips.xml', help='Original Input pedestrian trips file')
    parser.add_argument('--pedestrian_output_trips', type=str, default='./scaled_pedtrips.xml', help='Output pedestrian trips file')

    # If required to manually scale the demand (this happens automatically every episode as part of reset).
    parser.add_argument('--manual_scale_demand', type=bool, default=False, help='Manually scale demand before starting the simulation')
    parser.add_argument('--manual_scale_factor', type=float, default=3.0, help='Manual demand scaling factor (default: 1.0)')
    parser.add_argument('--demand_scale_min', type=float, default=0.5, help='Minimum demand scaling factor (default: 0.5)')
    parser.add_argument('--demand_scale_max', type=float, default=5.0, help='Maximum demand scaling factor (default: 5.0)')

    # PPO
    #parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available (default: use CPU)')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total number of timesteps to train (default: 1000000)')
    parser.add_argument('--max_timesteps', type=int, default=2500, help='Maximum number of steps in one episode (default: 500)')
    parser.add_argument('--update_freq', type=int, default=512, help='Number of timesteps between each policy update (default: 2048)')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--K_epochs', type=int, default=4, help='Number of epochs to update policy (default: 4)')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter for PPO (default: 0.2)')
    parser.add_argument('--save_freq', type=int, default=10, help='Save model every n episodes (default: 10, 0 to disable)')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient (default: 0.5)')

    args = parser.parse_args()
    main(args)