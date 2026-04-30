import numpy as np
import torch
from itertools import product
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Global configuration (match paper §4.2-4.3)
ACTION_SPACE = np.linspace(-2.0, 2.0, 201, dtype=np.float32)
ACTION_SPACE_TENSOR = torch.tensor(ACTION_SPACE, dtype=torch.float32).unsqueeze(1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"SKVI Using device: {device}")
ACTION_SPACE_TENSOR = ACTION_SPACE_TENSOR.to(device)

# Random seed
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

# State dictionary function φ(x): monomial feature mapping
def phi(x, order=2):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().to(device)
    else:
        x = x.float().to(device)
    
    x = x.reshape(-1, x.shape[-1])
    d_x = x.shape[1]
    
    indices = []
    for o in range(1, order+1):
        for idx in product(range(d_x), repeat=o):
            if list(idx) == sorted(idx):
                indices.append(idx)
    
    features = []
    for idx in indices:
        feat = torch.prod(x[:, idx], dim=1, keepdim=True)
        features.append(feat)
    
    phi_x = torch.cat(features, dim=1)
    return phi_x.squeeze() if phi_x.shape[0] == 1 else phi_x

# Action dictionary function ψ(u): monomial feature mapping
def psi(u, order=1):
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u).float().to(device)
    else:
        u = u.float().to(device)
    
    u = u.reshape(-1, u.shape[-1])
    d_u = u.shape[1]
    
    indices = []
    for o in range(1, order+1):
        for idx in product(range(d_u), repeat=o):
            if list(idx) == sorted(idx):
                indices.append(idx)
    
    features = []
    for idx in indices:
        feat = torch.prod(u[:, idx], dim=1, keepdim=True)
        features.append(feat)
    
    psi_u = torch.cat(features, dim=1)
    return psi_u.squeeze() if psi_u.shape[0] == 1 else psi_u

# Compute K^u matrix from Koopman tensor and action features
def get_Ku(koopman_M, psi_u):
    D_x = koopman_M.shape[0]
    D_u = psi_u.shape[-1]
    M_3d = koopman_M.reshape(D_x, D_x, D_u)
    Ku = torch.einsum('b u, x y u -> b x y', psi_u, M_3d)
    return Ku

class SKVIAgent:
    def __init__(self, data, alpha, epsilon,
                 batch_size, state_dim, action_dim, ref_point,
                 koopman_M, state_cost_matrix, action_cost_matrix,
                 phi_order=2, psi_order=1):
        self.data = data
        self.alpha = alpha
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.phi_order = phi_order
        self.psi_order = psi_order
        self.current_w = None
        
        # Move Koopman matrix to GPU
        if isinstance(koopman_M, np.ndarray):
            self.koopman_M = torch.from_numpy(koopman_M).float().to(device)
        else:
            self.koopman_M = koopman_M.float().to(device)
        
        # Precompute psi features and Ku matrix for action space
        PSI_ACTION_SPACE = psi(ACTION_SPACE_TENSOR, order=self.psi_order)
        self.Ku_action_space = get_Ku(self.koopman_M, PSI_ACTION_SPACE)
        self.Q = torch.tensor(state_cost_matrix, dtype=torch.float32).to(device)
        self.R = torch.tensor(action_cost_matrix, dtype=torch.float32).to(device)
        self.x_e = torch.tensor(ref_point, dtype=torch.float32).to(device)

    # Batch version of soft policy calculation
    def soft_policy_batch(self, x_batch):
        batch_size = x_batch.shape[0]
        x_batch_expand = x_batch.unsqueeze(1)
        u_expand = ACTION_SPACE_TENSOR.unsqueeze(0)
        
        # Compute state cost and action cost
        x_quad = torch.einsum('b i, i j, b j -> b', x_batch - self.x_e, self.Q, x_batch - self.x_e)
        x_quad = x_quad.unsqueeze(1)
        u_quad = torch.einsum('a i, i j, a j -> a', ACTION_SPACE_TENSOR, self.R, ACTION_SPACE_TENSOR)
        u_quad = u_quad.unsqueeze(0)
        c = x_quad + u_quad
        
        # Compute w^T K^u φ(x)
        phi_x_batch = phi(x_batch, self.phi_order)
        if phi_x_batch.dim() == 1:
            phi_x_batch = phi_x_batch.unsqueeze(0)
        w = self.current_w
        Ku_phi = torch.einsum('a x y, b y -> b a x', self.Ku_action_space, phi_x_batch)
        w_term = torch.einsum('x, b a x -> b a', w, Ku_phi)
        
        # Compute soft policy
        numerators = torch.exp(-(c + w_term) / self.alpha)
        Z_x = torch.sum(numerators, dim=1, keepdim=True) + 1e-10
        pi_star = numerators / Z_x
        
        return pi_star

    # Batch version of expected value calculation
    def compute_expected_value_batch(self, x_batch, pi_star_batch):
        batch_size = x_batch.shape[0]
        
        # Compute state and action cost
        x_quad = torch.einsum('b i, i j, b j -> b', x_batch - self.x_e, self.Q, x_batch - self.x_e)
        x_quad = x_quad.unsqueeze(1)
        u_quad = torch.einsum('a i, i j, a j -> a', ACTION_SPACE_TENSOR, self.R, ACTION_SPACE_TENSOR)
        u_quad = u_quad.unsqueeze(0)
        c = x_quad + u_quad
        
        # Compute log(pi_star)
        log_pi = torch.log(pi_star_batch + 1e-10)
        
        # Compute w^T K^u φ(x)
        phi_x_batch = phi(x_batch, self.phi_order)
        if phi_x_batch.dim() == 1:
            phi_x_batch = phi_x_batch.unsqueeze(0)
        Ku_phi = torch.einsum('a x y, b y -> b a x', self.Ku_action_space, phi_x_batch)
        w_term = torch.einsum('x, b a x -> b a', self.current_w, Ku_phi)
        
        # Compute target term and expected value
        target_term = c + self.alpha * log_pi + w_term
        expected_value = torch.sum(pi_star_batch * target_term, dim=1)
        
        return expected_value

    # Optimized SKVI training with full batching
    def train_skvi(self):
        # Initialize value function parameter w
        w = torch.normal(0, 1, (self.koopman_M.shape[0],), device=device, dtype=torch.float32)
        self.current_w = w
        abe = float('inf')
        
        # Precompute phi features for all data
        X = np.array([x for x, _, _ in self.data])
        X_tensor = torch.from_numpy(X).float().to(device)
        Phi = phi(X_tensor, self.phi_order)
        
        # Iterative optimization of w
        iter_count = 0
        while abe > self.epsilon and iter_count < 100:
            iter_count += 1
            
            # Batch sampling
            rng = torch.Generator(device=device)
            rng.manual_seed(42 + iter_count)
            indices = torch.randint(0, X_tensor.shape[0], (self.batch_size,), generator=rng, device=device)
            batch_data = X_tensor[indices]
            
            # Compute y in batch
            pi_star_batch = self.soft_policy_batch(batch_data)
            y = self.compute_expected_value_batch(batch_data, pi_star_batch)
            
            # OLS solution for w
            Phi_batch = Phi[indices]
            Phi_T = Phi_batch.T
            reg = 1e-6 * torch.eye(Phi_T.shape[0], dtype=torch.float32, device=device)
            w_new = torch.linalg.inv(Phi_T @ Phi_batch + reg) @ Phi_T @ y
            
            # Update w
            self.current_w = w_new
            w = w_new
            
            # Compute ABE
            pi_star_eval_batch = self.soft_policy_batch(batch_data)
            y_eval = self.compute_expected_value_batch(batch_data, pi_star_eval_batch)
            abe = torch.sum((Phi_batch @ w - y_eval)**2) / len(batch_data)
            abe_np = abe.cpu().item()
            print(f"Iteration {iter_count} | ABE：{abe_np:.6f}")
        
        print("SKVI training converged")
        w_np = w.cpu().numpy()
        return w_np

    # Optimal action selection based on trained w
    def get_action(self, state, w):
        # Convert state to tensor and move to GPU
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        else:
            state = state.float().to(device)
        
        # Set current_w temporarily
        if isinstance(w, np.ndarray):
            self.current_w = torch.from_numpy(w).float().to(device)
        else:
            self.current_w = w.float().to(device)
        
        # Compute policy and select action
        pi_star = self.soft_policy_batch(state.unsqueeze(0))
        action_idx = torch.argmax(pi_star, dim=1).cpu().item()
        return ACTION_SPACE[action_idx]