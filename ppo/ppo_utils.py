import torch
import torch.multiprocessing as mp
from torch_geometric.data import Batch

class WelfordNormalizer:
    def __init__(self, shape, eps=1e-8):
        """
        Online normalization (for both state and reward) using Welford's algorithm.
        Each lower level PPO actor uses its own copy of the old policy.
        However, a single (global) instance of the normalizer will be updated by all workers (as a shared resource).
        To avoid race conditions, lock is used. 
        """
        self.mean = torch.zeros(shape, dtype=torch.float32).share_memory_() 
        self.M2 = torch.zeros(shape, dtype=torch.float32).share_memory_() 
        self.count = mp.Value('i', 0) # A variable i that is shared between processes and is init to 0.
        self.eps = eps
        self.lock = mp.Lock()
        self.training = True # Only update the normalizer when training is True.

    def eval(self,):
        self.training = False

    def manual_load(self, mean, M2, count):
        """
        During evaluation, load the normalizer stats.
        """
        self.mean.copy_(mean)
        self.M2.copy_(M2)
        self.count.value = count

    def update(self, x):
        """
        Update running statistics using a new sample x.
        """
        with self.lock:
            if self.count.value == 0:
                # First sample: initialize mean and zero-out M2.
                self.mean.copy_(x)
                self.M2.zero_()
                self.count.value = 1
            else:
                self.count.value += 1
                delta = x - self.mean
                self.mean.add_(delta / self.count.value)
                delta2 = x - self.mean
                self.M2.add_(delta * delta2)

    def variance(self):
        with self.lock:
            if self.count.value < 2:
                # Not enough samples: return a tensor of ones with the same shape as mean
                return torch.ones_like(self.mean)
            else:
                return self.M2 / (self.count.value - 1)

    def std(self):
        return torch.sqrt(self.variance()) + self.eps

    def normalize(self, x):
        """
        Normalize the sample x using the running stats
        - x (torch.Tensor or array-like)
        """
        if self.training:
            self.update(x)
        return (x - self.mean) / self.std()
    

class Memory:
    """
    For saving experiences.
    Memories will be made in CPU but loaded in GPU for the policy updates.
    """
    def __init__(self,):
        self.states = []
        self.actions = []
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        
    def append(self, state, action, value, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action) 
        self.values.append(value) 
        self.logprobs.append(logprob)
        self.rewards.append(reward) # scalars
        self.is_terminals.append(done) # scalars


class GraphDataset(torch.utils.data.Dataset):
    """
    Graph data loading helper for the higher-level agent
    """
    def __init__(self, states, actions, logprobs, advantages, returns):
        self.states = states 
        self.actions = actions
        self.logprobs = logprobs
        self.advantages = advantages
        self.returns = returns

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.logprobs[idx],
            self.advantages[idx],
            self.returns[idx]
        )

def collate_fn(data):
    """
    Collate function for the higher-level agent
    """
    states_batch, actions_batch, old_logprobs_batch, advantages_batch, returns_batch = zip(*data)
    states_batch = Batch.from_data_list(states_batch)  

    # Concatenate actions_batch tensors along the first dimension
    actions_batch = torch.cat([a for a in actions_batch], dim=0)  # Now shape is (batch_size, max_proposals, 2)

    # Concatenate old_logprobs_batch tensors along the first dimension
    old_logprobs_batch = torch.cat([l for l in old_logprobs_batch], dim=0)  # Shape is (batch_size,)

    # Stack advantages and returns (already scalar tensors)
    advantages_batch = torch.stack(advantages_batch)  # Shape is (batch_size,)
    returns_batch = torch.stack(returns_batch)        # Shape is (batch_size,)

    return states_batch, actions_batch, old_logprobs_batch, advantages_batch, returns_batch