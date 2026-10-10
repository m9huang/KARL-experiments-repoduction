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
SEED = 5
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float64)

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
    def __init__(self, beta, state_dim, action_dim, fc1_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.q1 = nn.Linear(fc1_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        q1 = self.q1(x)
        return q1
    
LOG_STD_MIN = -5
LOG_STD_MAX = 2

# Actor network for action sampling
class ActorNetwork(nn.Module):
    def __init__(self, lr, state_dim, action_dim, fc1_dim, max_action=2):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.mu = nn.Linear(fc1_dim, action_dim)
        self.log_std = nn.Linear(fc1_dim, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.tiny_positive = 1e-6
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
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

# State dictionary function φ(x): monomial feature mapping
def phi(x, order=2):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).double().to(device)
    x = x.reshape(-1, x.shape[-1])
    d_x = x.shape[1]
    indices = []
    for o in range(1, order+1):
        for idx in product(range(d_x), repeat=o):
            if list(idx) == sorted(idx):
                indices.append(idx)
    features = [torch.ones(x.shape[0], 1, device=device, dtype=torch.float64)]
    for idx in indices:
        feat = torch.prod(x[:, idx], dim=1, keepdim=True)
        features.append(feat)
    phi_x = torch.cat(features, dim=1)
    return phi_x.squeeze() if phi_x.shape[0] == 1 else phi_x

# Action dictionary function ψ(u): monomial feature mapping
def psi(u, order=2):
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u).double().to(device)
    u = u.reshape(-1, u.shape[-1])
    d_u = u.shape[1]
    indices = []
    for o in range(1, order+1):
        for idx in product(range(d_u), repeat=o):
            if list(idx) == sorted(idx):
                indices.append(idx)
    features = [torch.ones(u.shape[0], 1, device=device, dtype=torch.float64)]
    for idx in indices:
        feat = torch.prod(u[:, idx], dim=1, keepdim=True)
        features.append(feat)
    psi_u = torch.cat(features, dim=1)
    return psi_u.squeeze() if psi_u.shape[0] == 1 else psi_u

# Train Koopman tensor M from collected data
def train_koopman_tensor(koopman_data, phi_order=2, psi_order=2):
    x_list, u_list, x_prime_list = [], [], []
    for x, u, x_prime in koopman_data:
        x_list.append(phi(x, phi_order).cpu().numpy())
        u_list.append(psi(u, psi_order).cpu().numpy())
        x_prime_list.append(phi(x_prime, phi_order).cpu().numpy())
    Phi = np.array(x_list)
    Psi = np.array(u_list)
    Phi_prime = np.array(x_prime_list)
    N = Phi.shape[0]
    D_x = Phi.shape[1]
    D_u = Psi.shape[1] if Psi.ndim > 1 else 1

    # Build Kronecker product matrix
    Psi_kronecker_Phi = np.zeros((N, D_x * D_u))
    for i in range(N):
        Psi_kronecker_Phi[i] = np.kron(Psi[i], Phi[i])

    # Linear regression with L2 regularization
    # lambda_reg = 1e-6
    # A = Psi_kronecker_Phi.T @ Psi_kronecker_Phi + lambda_reg * np.eye(D_x * D_u)
    # B = Phi_prime.T @ Psi_kronecker_Phi
    # Koopman_M = np.linalg.solve(A.T, B.T).T
    Koopman_M, _, _, _ = np.linalg.lstsq(Psi_kronecker_Phi, Phi_prime, rcond=None)
    Koopman_M = Koopman_M.T

    # Convert to torch tensor
    Koopman_M = torch.from_numpy(Koopman_M).double().to(device)
    return Koopman_M

# Compute action-dependent Koopman matrix K^u
def get_Ku(koopman_M, u, phi_order=2, psi_order=2):
    psi_u = psi(u, psi_order)
    D_x = koopman_M.shape[0]
    D_u = psi_u.shape[1]
    batch_size = psi_u.shape[0]
    M_3d = koopman_M.reshape(D_x, D_u, D_x)
    Ku = torch.einsum('b u, x u y -> b x y', psi_u, M_3d)
    return Ku

# SAKC Agent (replaces original SAC Agent)
class SAKCAgent:
    def __init__(self, state_dim, action_dim, memo_capacity,
                 lr, alpha, autotune, alpha_lr, v_lr, q_lr, gamma, tau, layer1_dim, max_action, batch_size,
                 policy_freq, koopman_M, phi_order=2, psi_order=2):
        self.alpha = alpha
        self.autotune = autotune
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.phi_order = phi_order
        self.psi_order = psi_order
        self.policy_freq = policy_freq
        self.koopman_M = koopman_M
        D_x = phi(torch.randn(1, state_dim).to(device), phi_order).shape[-1]
        self.D_x = D_x

        # Experience replay
        self.memory = ReplayMemory(memo_capacity, state_dim, action_dim)

        # Actor/Critic networks
        self.actor = ActorNetwork(lr, state_dim, action_dim, layer1_dim, max_action).to(device)
        self.critic_1 = CriticNetwork(q_lr, state_dim, action_dim, layer1_dim).to(device)
        self.critic_2 = CriticNetwork(q_lr, state_dim, action_dim, layer1_dim).to(device)

        # Koopman value function parameters
        self.w = torch.nn.Parameter(torch.randn(D_x).double().to(device), requires_grad=True)
        self.w_optimizer = optim.Adam([self.w], lr=v_lr)
        self.target_w = torch.randn(D_x).double().to(device)
        self.target_w.data = self.w.data.clone()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim

    # Get action from actor network
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float64).to(device)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        return action.cpu().detach().numpy()

    # Add experience to replay memory
    def add_memo(self, state, action, reward, new_state, done):
        self.memory.add_memory(state, action, reward, new_state, done)

    # Core update method for SAKC
    def update(self, actor_update=True):
        if self.memory.memo_counter < self.batch_size:
            return
        
        # Sample batch from replay memory
        state, action, reward, next_state, done = self.memory.sample_memory(self.batch_size)
        state = torch.tensor(state, dtype=torch.float64).to(device)
        action = torch.tensor(action, dtype=torch.float64).to(device)
        reward = torch.tensor(reward, dtype=torch.float64).to(device).view(-1, 1)
        done = torch.tensor(done, dtype=torch.bool).to(device).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float64).to(device)

        # Compute Koopman value function V_w(x)
        phi_x = phi(state, self.phi_order)
        V_w = torch.matmul(phi_x, self.w).view(-1, 1)

        # Update Koopman value weight w
        self.w_optimizer.zero_grad()
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        q1 = self.critic_1.forward(state, actions)
        q2 = self.critic_2.forward(state, actions)
        q_min = torch.min(q1, q2)
        v_target = q_min - self.alpha * log_probs
        w_loss = 0.5 * F.mse_loss(V_w, v_target.detach())
        w_loss.backward()
        # torch.nn.utils.clip_grad_norm_([self.w], max_norm=5.0)
        self.w_optimizer.step()

        # Soft update target weight
        with torch.no_grad():
            self.target_w.data = self.tau * self.w.data + (1 - self.tau) * self.target_w.data

        # Update actor network
        if actor_update:
            for _ in range(self.policy_freq):
                self.actor.optimizer.zero_grad()
                actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
                q1 = self.critic_1.forward(state, actions)
                q2 = self.critic_2.forward(state, actions)
                q_min = torch.min(q1, q2)
                actor_loss = torch.mean(self.alpha * log_probs - q_min)
                actor_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
                self.actor.optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi_alpha = self.actor.sample_normal(state, reparameterize=False)

                    alpha_loss = (-self.log_alpha * (log_pi_alpha + self.target_entropy)).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()

                    self.alpha = self.log_alpha.exp().item()

        # Update critic network with Koopman target Q
        with torch.no_grad():
            Ku = get_Ku(self.koopman_M, action, self.phi_order, self.psi_order)
            phi_x = phi(state, self.phi_order).unsqueeze(-1)
            Ku_phi_x = torch.bmm(Ku, phi_x).squeeze(-1)
            next_V = torch.matmul(Ku_phi_x, self.target_w).view(-1, 1)
            target_Q = reward + self.gamma * next_V * (1.0 - done.float())
            # target_Q = reward + self.gamma * torch.matmul(Ku_phi_x, self.target_w).view(-1, 1)
            # target_Q[done] = 0.0

        # Update critic 1
        self.critic_1.optimizer.zero_grad()
        q1_pred = self.critic_1.forward(state, action)
        critic1_loss = 0.5 * F.mse_loss(q1_pred, target_Q)
        critic1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=5.0)
        self.critic_1.optimizer.step()

        # Update critic 2
        self.critic_2.optimizer.zero_grad()
        q2_pred = self.critic_2.forward(state, action)
        critic2_loss = 0.5 * F.mse_loss(q2_pred, target_Q)
        critic2_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=5.0)
        self.critic_2.optimizer.step()