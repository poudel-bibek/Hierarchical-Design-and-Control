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
    """
    Define the neural networks for both actor and critic
    """
    def __init__(self, state_dim, action_dim, device):
        super(ActorCritic, self).__init__()
        self.device = device
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
    def act(self, state):
        """
        Select an action based on the current state
        """
        state = state.to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, states, actions):
        """
        Evaluate the actions given the states
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip, device):
        """
        Initialize PPO agent
        """
        self.device = device
        self.policy = ActorCritic(state_dim, action_dim, device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
    def update(self, memory):
        """
        Update the policy network
        """
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert list to tensor
        old_states = torch.stack(memory.states).detach().to(self.device)
        old_actions = torch.stack(memory.actions).detach().to(self.device)
        old_logprobs = torch.stack(memory.logprobs).detach().to(self.device)
        
        losses = []
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*nn.MSELoss()(state_values, rewards) - 0.01*dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            losses.append(loss.mean().item())
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        return np.mean(losses)

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
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim, args.lr, args.gamma, args.K_epochs, args.eps_clip, device=device)
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

    for i_episode in range(1, args.num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        for t in range(args.max_timesteps):
            state_tensor = torch.FloatTensor(state).to(device)
            action, log_prob = ppo.policy_old.act(state_tensor)
            
            next_state, reward, done, _, _ = env.step(action)
            
            # Saving experience in memory
            memory.append(state, action, log_prob, reward, done)
            
            state = next_state
            episode_reward += reward

            if done:
                break
        
        # Update PPO and get loss
        loss = ppo.update(memory)
        memory.clear_memory()
        
        # TensorBoard logging
        writer.add_scalar('Reward/Episode', episode_reward, i_episode)
        writer.add_scalar('Loss/Episode', loss, i_episode)
        writer.add_scalar('Episode Length', t+1, i_episode)
        
        # Logging
        if i_episode % 1 == 0:
            print(f'Episode {i_episode} \t Length: {t+1} \t Reward: {episode_reward:.2f} \t Loss: {loss:.4f}')
        
        # Save model periodically
        if args.save_freq > 0 and i_episode % args.save_freq == 0:
            torch.save(ppo.policy.state_dict(), os.path.join(save_dir, f'model_episode_{i_episode}.pth'))
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(ppo.policy.state_dict(), os.path.join(save_dir, 'best_model.pth'))
    
    env.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation with PPO.')

    # Simulation
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI (default: False)')
    parser.add_argument('--step_length', type=float, default=1.0, help='Simulation step length (default: 1.0)') # What is one unit of increment in the simulation?
    parser.add_argument('--auto_start', action='store_true', default=True, help='Automatically start the simulation')
    parser.add_argument('--input_trips', type=str, default='./original_vehtrips.xml', help='Original Input trips file')
    parser.add_argument('--output_trips', type=str, default='./scaled_vehtrips.xml', help='Output trips file')

    # If required to manually scale the demand (this happens automatically every episode as part of reset).
    parser.add_argument('--manual_scale_demand', type=bool, default=False, help='Manually scale demand before starting the simulation')
    parser.add_argument('--manual_scale_factor', type=float, default=3.0, help='Manual demand scaling factor (default: 1.0)')
    parser.add_argument('--demand_scale_min', type=float, default=0.5, help='Minimum demand scaling factor (default: 0.5)')
    parser.add_argument('--demand_scale_max', type=float, default=5.0, help='Maximum demand scaling factor (default: 5.0)')

    # PPO
    #parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available (default: use CPU)')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to run (default: 1000)')
    parser.add_argument('--max_timesteps', type=int, default=500, help='Maximum number of steps in one episode (default: 2500)')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--K_epochs', type=int, default=4, help='Number of epochs to update policy (default: 4)')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter for PPO (default: 0.2)')
    parser.add_argument('--save_freq', type=int, default=2, help='Save model every n episodes (default: 2, 0 to disable)')

    args = parser.parse_args()
    main(args)