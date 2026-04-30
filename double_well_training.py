import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium
from gymnasium import spaces
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter1d
from sakc_agent import SAKCAgent, train_koopman_tensor, phi, psi
from skvi_agent import SKVIAgent

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

class DoubleWellEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.state_dim = 2
        self.action_dim = 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(self.action_dim,), dtype=np.float32)
        self.state = None
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.action_dim)
        self.x_e = np.zeros(self.state_dim, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.normal(0, 1, size=self.state_dim).astype(np.float32)
        return self.state, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_val = np.squeeze(action).item()
        
        term = np.array([[0.7, self.state[0]], [0, 0.5]], dtype=np.float32) @ np.random.normal(0, 1, size=self.state_dim).astype(np.float32)
        next_state = np.array([4*self.state[0] - 4*self.state[0]**3 + action_val,
                               -2*self.state[1] + action_val], dtype=np.float32) + term
        cost = self.state.T @ self.Q @ self.state + action.T @ self.R @ action
        reward = -cost
        done = np.any(np.abs(next_state) > 10.0)
        self.state = next_state
        return next_state, reward, done, False, {}

# Hyperparameters initialization
scenario = "DoubleWell-KARL"
env = DoubleWellEnv()
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MEMORY_SIZE = 1000000
ALPHA_SAKC = 3e-4
ALPHA_SKVI = 0.1
BETA = 1e-3
GAMMA = 0.99
TAU = 0.005
EPSILON = 1e-6
LAYER1_DIM = 64
LAYER2_DIM = 64
SAKC_BATCH_SIZE = 256
SKVI_BATCH_SIZE = 8192
NUM_EPISODE = 200
NUM_STEP = 200
KOOPMAN_DATA_SIZE = 30000
PHI_ORDER = 2
PSI_ORDER = 2

# Path and log initialization
current_path = os.path.dirname(os.path.realpath(__file__))
sakc_model_dir = current_path + '/double_well_models_sakc/'
skvi_model_dir = current_path + '/double_well_models_skvi/'
reward_dir = current_path + '/double_well_rewards/'
os.makedirs(sakc_model_dir, exist_ok=True)
os.makedirs(skvi_model_dir, exist_ok=True)
os.makedirs(reward_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d%H%M%S")
SAKC_best_reward = -np.inf
SKVI_best_reward = -np.inf
SAKC_REWARD_BUFFER = []
SKVI_REWARD_BUFFER = []
PLOT_REWARD = True

# Pre-train Koopman tensor with random policy
print("===== Pre-training Koopman Tensor with random policy =====")
koopman_data = []
rand_env = DoubleWellEnv()
for _ in range(KOOPMAN_DATA_SIZE):
    state = rand_env.reset(SEED)[0]
    action = rand_env.action_space.sample()
    next_state, _, done, _, _ = rand_env.step(action)
    koopman_data.append((state, action, next_state))
    if done:
        rand_env.reset(SEED)
Koopman_M = train_koopman_tensor(koopman_data, phi_order=PHI_ORDER, psi_order=PSI_ORDER)
print("===== Koopman Tensor Pre-training Done =====")

# Initialize SAKC and SKVI agents
agent_1 = SAKCAgent(
    state_dim=STATE_DIM, action_dim=ACTION_DIM, memo_capacity=MEMORY_SIZE,
    alpha=ALPHA_SAKC, beta=BETA, gamma=GAMMA, tau=TAU,
    layer1_dim=LAYER1_DIM, layer2_dim=LAYER2_DIM, batch_size=SAKC_BATCH_SIZE,
    koopman_M=Koopman_M, phi_order=PHI_ORDER, psi_order=PSI_ORDER
)

agent_2 = SKVIAgent(
    data=koopman_data, alpha=ALPHA_SKVI, epsilon=EPSILON, batch_size=SKVI_BATCH_SIZE, 
    state_dim=STATE_DIM, action_dim=ACTION_DIM, ref_point=env.x_e,
    koopman_M=Koopman_M, state_cost_matrix=env.Q, action_cost_matrix=env.R,
    phi_order=PHI_ORDER, psi_order=PSI_ORDER
)

# SAKC training loop
print("===== Start SAKC Training on Linear System =====")
for episode_i in range(NUM_EPISODE):
    state, others = env.reset(SEED)
    episode_reward = 0
    for step_i in range(NUM_STEP):
        action = agent_1.get_action(state)
        next_state, reward, done, trunc, others = env.step(action)
        agent_1.add_memo(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
        agent_1.update()
        if done:
            break
    SAKC_REWARD_BUFFER.append(episode_reward)
    avg_reward = np.mean(SAKC_REWARD_BUFFER[-100:])
    if avg_reward > SAKC_best_reward:
        SAKC_best_reward = avg_reward
        torch.save(agent_1.actor.state_dict(), sakc_model_dir + f'sakc_actor_{timestamp}.pth')
        torch.save(agent_1.w, sakc_model_dir + f'sakc_w_{timestamp}.pth')
        print(f'... saving SAKC model with best avg reward: {SAKC_best_reward:.1f}...')
    print(f'Episode {episode_i:3d}, reward {episode_reward:.1f}, avg_reward {avg_reward:.1f}')

# SKVI training loop
print("===== Start SKVI Training on Linear System =====")
w_opt = agent_2.train_skvi()
torch.save(w_opt, skvi_model_dir + f'skvi_w_{timestamp}.pth')
for episode_i in range(NUM_EPISODE):
    state, others = env.reset(SEED)
    episode_reward = 0
    for step_i in range(NUM_STEP):
        action = agent_2.get_action(state, w_opt)
        next_state, reward, done, trunc, others = env.step(action)
        episode_reward += reward
        state = next_state
        if done:
            break
    
    SKVI_REWARD_BUFFER.append(episode_reward)
    avg_reward = np.mean(SKVI_REWARD_BUFFER[-500:])
    print(f'Episode {episode_i:3d}, reward {episode_reward:.1f}, avg_reward {avg_reward:.1f}')

env.close()

# Save reward data and plot curves
np.savetxt(reward_dir + f'/sakc_reward_{scenario}_{timestamp}.txt', SAKC_REWARD_BUFFER)
np.savetxt(reward_dir + f'/skvi_reward_{scenario}_{timestamp}.txt', SKVI_REWARD_BUFFER)
if PLOT_REWARD:
    plt.figure(figsize=(10, 6))
    steps = np.arange(len(SKVI_REWARD_BUFFER)) * NUM_STEP
    plt.plot(steps, SAKC_REWARD_BUFFER, color='purple', alpha=0.5, label='SAKC Reward')
    plt.plot(steps, SKVI_REWARD_BUFFER, color='orange', alpha=0.5, label='SKVI Reward')
    plt.plot(steps, gaussian_filter1d(SAKC_REWARD_BUFFER, sigma=5), color='purple', linewidth=2, label='SAKC Smoothed')
    plt.plot(steps, gaussian_filter1d(SKVI_REWARD_BUFFER, sigma=5), color='orange', linewidth=2, label='SKVI Smoothed')
    plt.title(f'SAKC and SKVI Rewards on {scenario} (KARL Paper Figure6)', fontsize=14)
    plt.xlabel('Total Steps in Environment', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"SAKC-and-SKVI-Rewards-{scenario}-{timestamp}.png", format='png', dpi=300)
    plt.show()

print("===== Training Finished =====")