import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gymnasium
from gymnasium import spaces
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from control import lqr
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import solve_ivp
from sakc_agent import SAKCAgent, train_koopman_tensor
from skvi_agent import SKVIAgent
from sac_v_agent import SAC_V_Agent
from sac_q_agent import SAC_Q_Agent

SEED = 5
np.random.seed(SEED)
torch.manual_seed(SEED)

class FluidFlowEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.mu = 0.1
        self.omega = 1.0
        self.A = -0.1
        self.Lambda = 1
        self.state_dim = 3
        self.action_dim = 1
        self.Jacobian_x = np.array([[0.1, -1.0, 0.0], [1.0, 0.1, 0.0], [0.0, 0.0, -1.0]])
        self.Jacobian_u = np.array([[0.0], [1.0], [0.0]])
        self.max_state = 10.0
        self.max_action = 10.0
        self.observation_space = spaces.Box(low=-self.max_state, high=self.max_state, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_action, high=self.max_action, shape=(self.action_dim,), dtype=np.float32)
        self.state = None
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.action_dim)
        self.x_e = np.zeros(self.state_dim, dtype=np.float32)
        self.dt = 0.01
        # P = la.solve_continuous_are(self.Jacobian_x, self.Jacobian_u, self.Q, self.R)
        # self.K = la.inv(self.R) @ self.Jacobian_u.T @ P
        self.K, _, _ = lqr(self.Jacobian_x, self.Jacobian_u, self.Q, self.R)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-1.0, 1.0, size=self.state_dim).astype(np.float32)
        self.step_count = 0
        return self.state, {}

    def continuous_dynamics(self, action):
        def dx_dt(t, state):
            f = np.array([
                self.mu * state[0] - self.omega * state[1] + self.A * state[0] * state[2],
                self.omega * state[0] + self.mu * state[1] + self.A * state[1] * state[2] + action,
                -self.Lambda * (state[2] - state[0]**2 - state[1]**2)
            ], dtype=np.float32)
            return f
        return dx_dt

    def f(self, state, action):
        sol = solve_ivp(self.continuous_dynamics(action), [0, self.dt], state, method='RK45')
        return sol.y[:, -1]

    def step(self, action):
        if torch.is_tensor(action):
            action = action.cpu().detach().numpy()
        action = np.clip(action, -self.max_action, self.max_action)
        action_val = np.squeeze(action).item()
        
        # dx_dt = np.array([self.mu*self.state[0] - self.omega*self.state[1] + self.A*self.state[0]*self.state[2],
        #                        self.omega*self.state[0] + self.mu*self.state[1] + self.A*self.state[1]*self.state[2] + action_val,
        #                        -self.Lambda*(self.state[2] - self.state[0]**2 - self.state[1]**2)], dtype=np.float32)
        # next_state = self.state + dx_dt * self.dt
        next_state = self.f(self.state, action_val)

        cost = self.state.T @ self.Q @ self.state + action.T @ self.R @ action
        reward = -cost
        done = self.step_count >= NUM_STEP
        self.state = next_state
        self.step_count += 1
        return next_state, reward, done, False, {}

# Hyperparameters initialization
scenario = "FluidFlow-KARL"
env = FluidFlowEnv()
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MEMORY_SIZE = 1000000
ALPHA_SAKC = 0.01
ALPHA_SKVI = 0.1
ACTOR_LR = 3e-4
BETA = 1e-3
GAMMA = 0.99
TAU = 0.005
EPSILON = 1e-6
LAYER1_DIM = 256
LAYER2_DIM = 256
SAKC_BATCH_SIZE = 256
SKVI_BATCH_SIZE = 8192
NUM_EPISODE = 200
NUM_STEP = 250
KOOPMAN_DATA_SIZE = 30000
PHI_ORDER = 2
PSI_ORDER = 2

# Path and log initialization
current_path = os.path.dirname(os.path.realpath(__file__))
sakc_model_dir = current_path + '/fluid_flow_models_sakc/'
skvi_model_dir = current_path + '/fluid_flow_models_skvi/'
sacv_model_dir = current_path + '/fluid_flow_models_sacv/'
sacq_model_dir = current_path + '/fluid_flow_models_sacq/'
reward_dir = current_path + '/fluid_flow_rewards/'
os.makedirs(sakc_model_dir, exist_ok=True)
os.makedirs(skvi_model_dir, exist_ok=True)
os.makedirs(sacv_model_dir, exist_ok=True)
os.makedirs(sacq_model_dir, exist_ok=True)
os.makedirs(reward_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d%H%M%S")
SAKC_best_reward = -np.inf
SKVI_best_reward = -np.inf
SACV_best_reward = -np.inf
SACQ_best_reward = -np.inf
LQR_REWARD_BUFFER = []
SAKC_REWARD_BUFFER = []
SKVI_REWARD_BUFFER = []
SACV_REWARD_BUFFER = []
SACQ_REWARD_BUFFER = []
PLOT_REWARD = True

# Pre-train Koopman tensor with random policy
print("===== Pre-training Koopman Tensor with random policy =====")
koopman_data = []
replay_buffer_data = []
rand_env = FluidFlowEnv()
state = rand_env.reset()[0]
for _ in range(KOOPMAN_DATA_SIZE):
    action = rand_env.action_space.sample()
    next_state, reward, done, _, _ = rand_env.step(action)
    koopman_data.append((state, action, next_state))
    replay_buffer_data.append((state, action, reward, next_state, done))
    if done:
        state = rand_env.reset()[0]
    else:
        state = next_state
Koopman_M = train_koopman_tensor(koopman_data, phi_order=PHI_ORDER, psi_order=PSI_ORDER)
print("===== Koopman Tensor Pre-training Done =====")

# Initialize SAKC agents
agent_1 = SAKCAgent(
    state_dim=STATE_DIM, action_dim=ACTION_DIM, memo_capacity=MEMORY_SIZE,
    lr=ACTOR_LR, alpha=ALPHA_SAKC, beta=BETA, gamma=GAMMA, tau=TAU,
    layer1_dim=LAYER1_DIM, layer2_dim=LAYER2_DIM, max_action=env.max_action, batch_size=SAKC_BATCH_SIZE,
    koopman_M=Koopman_M, phi_order=PHI_ORDER, psi_order=PSI_ORDER
)

# Initialize SKVI agent
agent_2 = SKVIAgent(
    data=koopman_data, alpha=ALPHA_SKVI, gamma=GAMMA, epsilon=EPSILON,
    batch_size=SKVI_BATCH_SIZE, state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=env.max_action, ref_point=env.x_e,
    koopman_M=Koopman_M, state_cost_matrix=env.Q, action_cost_matrix=env.R,
    phi_order=PHI_ORDER, psi_order=PSI_ORDER, dt=env.dt
)

# Initialize SAC(V) agent
agent_3 = SAC_V_Agent(
    state_dim=STATE_DIM, action_dim=ACTION_DIM, memo_capacity=MEMORY_SIZE,
    lr=ACTOR_LR, alpha=ALPHA_SAKC, beta=BETA, gamma=GAMMA, tau=TAU,
    layer1_dim=LAYER1_DIM, layer2_dim=LAYER2_DIM, max_action=env.max_action, batch_size=SAKC_BATCH_SIZE
)

# Initialize SAC(Q) agent
agent_4 = SAC_Q_Agent(
    state_dim=STATE_DIM, action_dim=ACTION_DIM, memo_capacity=MEMORY_SIZE,
    lr=ACTOR_LR, alpha=ALPHA_SAKC, beta=BETA, gamma=GAMMA, tau=TAU,
    layer1_dim=LAYER1_DIM, layer2_dim=LAYER2_DIM, max_action=env.max_action, batch_size=SAKC_BATCH_SIZE
)

# LQR
print("===== Start LQR on Linear System =====")
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0
    for step_i in range(NUM_STEP):
        action = -env.K @ state
        next_state, reward, done, trunc, others = env.step(action)
        episode_reward += reward
        state = next_state
        if done:
            break
    
    LQR_REWARD_BUFFER.append(episode_reward)
    avg_reward = np.mean(LQR_REWARD_BUFFER[-100:])
    print(f'Episode {episode_i:3d}, reward {episode_reward:.1f}, avg_reward {avg_reward:.1f}')

# SAKC training loop
print("===== Start SAKC Training on Linear System =====")
# Pre-fill SAKC replay buffer to prevent early training instability
# for s, a, r, s_next, d in replay_buffer_data[:5000]:
#     agent_1.add_memo(s, a, r, s_next, d)
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0
    for step_i in range(NUM_STEP):
        action = agent_1.get_action(state)
        if step_i % 25 == 0:
            print(f"SAKC Episode {episode_i}, Step {step_i}, State: {state}, Action: {action}")
        next_state, reward, done, trunc, others = env.step(action)
        agent_1.add_memo(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
        agent_1.update()
        if done:
            print(f"SAKC Episode {episode_i} terminated at step {step_i}, state: {state}")
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
    state, others = env.reset()
    episode_reward = 0
    for step_i in range(NUM_STEP):
        action = agent_2.get_action(state, w_opt)
        next_state, reward, done, trunc, others = env.step(action)
        episode_reward += reward
        state = next_state
        if done:
            print(f"SKVI Episode {episode_i} terminated at step {step_i}, state: {state}")
            break
    
    SKVI_REWARD_BUFFER.append(episode_reward)
    avg_reward = np.mean(SKVI_REWARD_BUFFER[-100:])
    print(f'Episode {episode_i:3d}, reward {episode_reward:.1f}, avg_reward {avg_reward:.1f}')

print("===== Start SAC(V) Training on Linear System =====")
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0
    for step_i in range(NUM_STEP):
        action = agent_3.get_action(state)
        if step_i % 25 == 0:
            print(f"SAC(V) Episode {episode_i}, Step {step_i}, State: {state}, Action: {action}")
        next_state, reward, done, trunc, others = env.step(action)
        agent_3.add_memo(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
        agent_3.update()
        if done:
            print(f"SAC(V) Episode {episode_i} terminated at step {step_i}, state: {state}")
            break
    SACV_REWARD_BUFFER.append(episode_reward)
    avg_reward = np.mean(SACV_REWARD_BUFFER[-100:])
    if avg_reward > SACV_best_reward:
        SACV_best_reward = avg_reward
        torch.save(agent_3.actor.state_dict(), sacv_model_dir + f'sacv_actor_{timestamp}.pth')
        print(f'... saving SAC(V) model with best avg reward: {SACV_best_reward:.1f}...')
    print(f'Episode {episode_i:3d}, reward {episode_reward:.1f}, avg_reward {avg_reward:.1f}')

print("===== Start SAC(Q) Training on Linear System =====")
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0
    for step_i in range(NUM_STEP):
        action = agent_4.get_action(state)
        if step_i % 25 == 0:
            print(f"SAC(Q) Episode {episode_i}, Step {step_i}, State: {state}, Action: {action}")
        next_state, reward, done, trunc, others = env.step(action)
        agent_4.add_memo(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
        agent_4.update()
        if done:
            print(f"SAC(Q) Episode {episode_i} terminated at step {step_i}, state: {state}")
            break
    SACQ_REWARD_BUFFER.append(episode_reward)
    avg_reward = np.mean(SACQ_REWARD_BUFFER[-100:])
    if avg_reward > SACQ_best_reward:
        SACQ_best_reward = avg_reward
        torch.save(agent_4.actor.state_dict(), sacq_model_dir + f'sacq_actor_{timestamp}.pth')
        print(f'... saving SAC(Q) model with best avg reward: {SACQ_best_reward:.1f}...')
    print(f'Episode {episode_i:3d}, reward {episode_reward:.1f}, avg_reward {avg_reward:.1f}')

env.close()

# Save reward data and plot curves
np.savetxt(reward_dir + f'/lqr_reward_{scenario}_{timestamp}_seed{SEED}.txt', LQR_REWARD_BUFFER)
np.savetxt(reward_dir + f'/sakc_reward_{scenario}_{timestamp}_seed{SEED}.txt', SAKC_REWARD_BUFFER)
np.savetxt(reward_dir + f'/skvi_reward_{scenario}_{timestamp}_seed{SEED}.txt', SKVI_REWARD_BUFFER)
np.savetxt(reward_dir + f'/sacv_reward_{scenario}_{timestamp}_seed{SEED}.txt', SACV_REWARD_BUFFER)
np.savetxt(reward_dir + f'/sacq_reward_{scenario}_{timestamp}_seed{SEED}.txt', SACQ_REWARD_BUFFER)
if PLOT_REWARD:
    plt.figure(figsize=(10, 6))
    steps = np.arange(len(SKVI_REWARD_BUFFER)) * NUM_STEP
    plt.plot(steps, SAKC_REWARD_BUFFER, color='purple', alpha=0.5, label='SAKC Reward')
    plt.plot(steps, SKVI_REWARD_BUFFER, color='orange', alpha=0.5, label='SKVI Reward')
    plt.plot(steps, LQR_REWARD_BUFFER, color='red', alpha=0.5, label='LQR Reward')
    plt.plot(steps, SACV_REWARD_BUFFER, color='blue', alpha=0.5, label='SAC(V) Reward')
    plt.plot(steps, SACQ_REWARD_BUFFER, color='green', alpha=0.5, label='SAC(Q) Reward')
    plt.plot(steps, gaussian_filter1d(SAKC_REWARD_BUFFER, sigma=5), color='purple', linewidth=2, label='SAKC Smoothed')
    plt.plot(steps, gaussian_filter1d(SKVI_REWARD_BUFFER, sigma=5), color='orange', linewidth=2, label='SKVI Smoothed')
    plt.plot(steps, gaussian_filter1d(LQR_REWARD_BUFFER, sigma=5), color='red', linewidth=2, label='LQR Smoothed')
    plt.plot(steps, gaussian_filter1d(SACV_REWARD_BUFFER, sigma=5), color='blue', linewidth=2, label='SAC(V) Smoothed')
    plt.plot(steps, gaussian_filter1d(SACQ_REWARD_BUFFER, sigma=5), color='green', linewidth=2, label='SAC(Q) Smoothed')
    plt.title(f'Rewards on {scenario} (KARL Paper Figure6)', fontsize=14)
    plt.xlabel('Total Steps in Environment', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Rewards-{scenario}-{timestamp}_seed{SEED}.png", format='png', dpi=300)
    plt.show()

print("===== Training Finished =====")