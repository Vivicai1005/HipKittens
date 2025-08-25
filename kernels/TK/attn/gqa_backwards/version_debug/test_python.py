import torch
import random
import math
import tk_kernel
import time

torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=3,        
    sci_mode=False,     
    linewidth=220,      
    threshold=float("inf")  
)

# **************************************************
# Reference
# **************************************************

def simple_flash_backward(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S - m.unsqueeze(-1)) / l.unsqueeze(-1)
    O = torch.matmul(P, V)

    # dV
    dV = torch.matmul(P.transpose(-2, -1), dO)

    # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)                 # (B, N, H, 1)
    dS = P * (torch.matmul(dO, V.transpose(-2, -1)) - Delta)   # (B, N, H, N)

    # chain rule through S = (Q K^T) * scale
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale

    return dQ, dK, dV, Delta

# **************************************************
# Generate inputs
# **************************************************


causal = False
b = 1
h = 1
n = 32
d = 32
dtype = torch.bfloat16
mean = 10
std = 0.1  

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

def generate_inputs():
    # Generate in BHND format (batch, heads, seq, dim) for reference
    Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda') 

    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    return Q, K, V, dO

# Generate base inputs in BHND format
Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = generate_inputs()

# **************************************************
# Tiled Reference
# **************************************************

Q_tiled = Q_bhnd.clone().contiguous().detach().requires_grad_(True)  
K_tiled = K_bhnd.clone().contiguous().detach().requires_grad_(True)  
V_tiled = V_bhnd.clone().contiguous().detach().requires_grad_(True)  
dO_tiled = dO_bhnd.clone().contiguous()  
QK = torch.matmul(Q_tiled.float(), K_tiled.transpose(-2, -1).float()) / math.sqrt(d)
m_tiled = QK.max(dim=-1, keepdim=True)[0] 
exp_scores = torch.exp(QK - m_tiled)  
l_tiled = exp_scores.sum(dim=-1, keepdim=True)  
P_tiled = exp_scores / l_tiled
O_tiled = torch.matmul(P_tiled, V_tiled.float())
m_tiled = m_tiled.squeeze(-1)
l_tiled = l_tiled.squeeze(-1)

dQ_tiled, dK_tiled, dV_tiled, delta_tiled = simple_flash_backward(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), m_tiled, l_tiled)
out_tiled_bhnd = O_tiled
q_grad_tiled_bhnd = dQ_tiled

# **************************************************
# ThunderKittens
# **************************************************

# Get forwards pass outputs
Q_tk = Q_bhnd.bfloat16().clone().contiguous().detach().requires_grad_(True)  
K_tk = K_bhnd.bfloat16().clone().contiguous().detach().requires_grad_(True)  
V_tk = V_bhnd.bfloat16().clone().contiguous().detach().requires_grad_(True)  
O_tk = O_tiled.bfloat16().clone()
dO_tk = dO_bhnd.float().clone()
m_tk = m_tiled.float().unsqueeze(-1)
l_tk = l_tiled.float().unsqueeze(-1)
print(m_tk.shape, l_tk.shape)

def test_dq(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S - m.unsqueeze(-1)) #/ l.unsqueeze(-1)
    O = torch.matmul(P, V)
    return P #m.unsqueeze(-1)

DQ = test_dq(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), m_tiled, l_tiled)
dQ_tk = torch.zeros_like(DQ)

# TK
tk_kernel.dispatch_bwd_combined(
    Q_tk,     
    K_tk,     
    V_tk,     
    O_tk,  

    dO_tk,    
    dQ_tk,   
    m_tk, 
    l_tk,
)

# TK vs PyTorch
num_print = 8
print("Gradient Q outputs:")
print("TK: ", dQ_tk[0, 0, 2:5, :num_print], "Max:", dQ_tk.max().item())
print("PyTorch: ", DQ[0, 0, 2:5, :num_print], "Max:", DQ.max().item())

