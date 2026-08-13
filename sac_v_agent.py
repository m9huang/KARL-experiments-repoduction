import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from itertools import product

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"SAKC Using device: {device}")

# Random seed
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

# Replay memory for experience storage
class ReplayMemory:
    def __init__(self, memo_capacity, state_dim, action_dim):
        self.memo_size = memo_capacity
        self.state_memo = np.zeros((self.memo_size, state_dim))
        self.new_state_memo = np.zeros((self.memo_size, state_dim))
        self.action_memo = np.zeros((self.memo_size, action_dim))
        self.reward_memo = np.zeros(self.memo_size)
        self.done_memo = np.zeros(self.memo_size)
        self.memo_counter = 0
        
    def add_memory(self, state, action, reward, new_state, done):
        index = self.memo_counter % self.memo_size
        self.state_memo[index] = state
        self.new_state_memo[index] = new_state
        self.action_memo[index] = action
        self.reward_memo[index] = reward
        self.done_memo[index] = done
        self.memo_counter += 1

    def sample_memory(self, batch_size):
        current_memo_size = min(self.memo_counter, self.memo_size)
        # rng = np.random.default_rng(SEED)
        # batch = rng.choice(current_memo_size, batch_size, replace=False)
        batch = np.random.choice(current_memo_size, batch_size, replace=False)
        batch_state = self.state_memo[batch]
        batch_next_state = self.new_state_memo[batch]
        batch_action = self.action_memo[batch]
        batch_reward = self.reward_memo[batch]
        batch_done = self.done_memo[batch]
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

# Critic network for Q-value estimation
class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q1 = nn.Linear(fc2_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q1 = self.q1(x)
        return q1
    
LOG_STD_MIN = -5
LOG_STD_MAX = 2

# Actor network for action sampling
class ActorNetwork(nn.Module):
    def __init__(self, lr, state_dim, action_dim, fc1_dim, fc2_dim, max_action=2):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.mu = nn.Linear(fc2_dim, action_dim)
        self.log_std = nn.Linear(fc2_dim, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.tiny_positive = 1e-6
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # mu = torch.tanh(self.mu(x)) * self.max_action
        mu = self.mu(x)
        log_std = self.log_std(x)
        # log_std = torch.clamp(log_std, min=-5, max=2)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        sigma = torch.exp(log_std)
        return mu, sigma

    def sample_normal(self, state, reparameterize):
        mu, sigma = self.forward(state)
        probability = Normal(mu, sigma)
        raw_action = probability.rsample() if reparameterize else probability.sample()
        tanh_action = torch.tanh(raw_action)
        scaled_actions = tanh_action * self.max_action
        log_prob = probability.log_prob(raw_action)
        log_prob -= torch.log(self.max_action * (1 - tanh_action.pow(2)) + self.tiny_positive)
        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(0)
        log_prob = log_prob.sum(1, keepdim=True)
        return scaled_actions, log_prob

# Value Network for state value estimation
class ValueNetwork(nn.Module):
    def __init__(self, beta, state_dim, fc1_dim, fc2_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.v = nn.Linear(fc2_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        return v

# SAKC Agent (replaces original SAC Agent)
class SAC_V_Agent:
    def __init__(self, state_dim, action_dim, memo_capacity,
                 lr, alpha, beta, gamma, tau, layer1_dim, layer2_dim, max_action, batch_size):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Experience replay
        self.memory = ReplayMemory(memo_capacity, state_dim, action_dim)

        # Actor/Critic networks
        self.actor = ActorNetwork(lr, state_dim, action_dim, layer1_dim, layer2_dim, max_action).to(device)
        self.critic_1 = CriticNetwork(beta, state_dim, action_dim, layer1_dim, layer2_dim).to(device)
        self.critic_2 = CriticNetwork(beta, state_dim, action_dim, layer1_dim, layer2_dim).to(device)
        self.value_net = ValueNetwork(beta, state_dim, layer1_dim, layer2_dim).to(device)
        self.target_value_net = ValueNetwork(beta, state_dim, layer1_dim, layer2_dim).to(device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        

    # Get action from actor network
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        return action.cpu().detach().numpy()

    # Add experience to replay memory
    def add_memo(self, state, action, reward, new_state, done):
        self.memory.add_memory(state, action, reward, new_state, done)

    # Core update method for SAKC
    def update(self):
        if self.memory.memo_counter < self.batch_size:
            return
        
        # Sample batch from replay memory
        state, action, reward, next_state, done = self.memory.sample_memory(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device).view(-1, 1)
        done = torch.tensor(done, dtype=torch.bool).to(device).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)

        # Update value network
        v_pred = self.value_net.forward(state)
        with torch.no_grad():
            actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
            q1 = self.critic_1.forward(state, actions)
            q2 = self.critic_2.forward(state, actions)
            q_min = torch.min(q1, q2)
            v_target = q_min - self.alpha * log_probs
        value_loss = 0.5 * F.mse_loss(v_pred, v_target)
        self.value_net.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=5.0)
        self.value_net.optimizer.step()

        # Update critic networks
        with torch.no_grad():
            next_v = self.target_value_net.forward(next_state)
            target_q = reward + self.gamma * next_v * (1.0 - done.float())
        
        # Update critic 1
        self.critic_1.optimizer.zero_grad()
        q1_pred = self.critic_1.forward(state, action)
        critic1_loss = 0.5 * F.mse_loss(q1_pred, target_q)
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=5.0)
        self.critic_1.optimizer.step()

        # Update critic 2
        self.critic_2.optimizer.zero_grad()
        q2_pred = self.critic_2.forward(state, action)
        critic2_loss = 0.5 * F.mse_loss(q2_pred, target_q)
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=5.0)
        self.critic_2.optimizer.step()

        # Update actor network
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        q1 = self.critic_1.forward(state, actions)
        q2 = self.critic_2.forward(state, actions)
        q_min = torch.min(q1, q2)
        actor_loss = torch.mean(self.alpha * log_probs - q_min)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor.optimizer.step()

        # Soft update target value network
        with torch.no_grad():
            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)