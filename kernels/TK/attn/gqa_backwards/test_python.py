import torch
import random
import math
import tk_kernel


use_aiter = True
if use_aiter:
    import aiter

torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=6,        
    sci_mode=False,     
    linewidth=220,      
    threshold=float("inf")  
)

# Parameters
causal = False
b = 1
h = 1
n = 32
d = 128
dtype = torch.bfloat16

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

def generate_inputs():
    mean = 5 #1e-1
    std = 0.1  # REDUCED from 10 to 0.1 for numerical stability
    
    # Generate in BHND format (batch, heads, seq, dim) for reference
    Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda') * 50

    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    return Q, K, V, dO

def reference_backwards(Q, K, V, dO, causal):
    """Reference implementation using BHND layout (batch, heads, seq, dim)"""
    # Convert to float64 and create new leaf tensors with requires_grad
    q_ = Q.detach().to(torch.float64).requires_grad_(True)
    k_ = K.detach().to(torch.float64).requires_grad_(True)
    v_ = V.detach().to(torch.float64).requires_grad_(True)
    dO_ = dO.to(torch.float64)
    
    # manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q_, k_.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_)
    
    output.backward(dO_)
    
    q_grad = q_.grad
    k_grad = k_.grad
    v_grad = v_.grad
    
    return output, q_grad, k_grad, v_grad

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

    return dQ, dK, dV

# Generate base inputs in BHND format
Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = generate_inputs()

# Clone for PyTorch reference (keep BHND format)
Q_pytorch = Q_bhnd.clone().detach().requires_grad_(True)
K_pytorch = K_bhnd.clone().detach().requires_grad_(True)
V_pytorch = V_bhnd.clone().detach().requires_grad_(True)
dO_pytorch = dO_bhnd.clone()

# Create leaf tensors for AITER (BNHD format)
Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
dO_aiter = dO_bhnd.transpose(1, 2).contiguous()  

# Create leaf tensors for Tiled (BNHD format)
Q_tiled = Q_bhnd.clone().contiguous().detach().requires_grad_(True)  
K_tiled = K_bhnd.clone().contiguous().detach().requires_grad_(True)  
V_tiled = V_bhnd.clone().contiguous().detach().requires_grad_(True)  
dO_tiled = dO_bhnd.clone().contiguous()  

# Create leaf tensors for TK (BNHD format)
Q_tk = Q_bhnd.bfloat16().clone().contiguous().detach().requires_grad_(True)  
K_tk = K_bhnd.bfloat16().clone().contiguous().detach().requires_grad_(True)  
V_tk = V_bhnd.bfloat16().clone().contiguous().detach().requires_grad_(True)  
dO_tk = dO_bhnd.bfloat16().clone().contiguous() 

# AITER forward and backward
if use_aiter:
    print("\nRunning AITER...")
    out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal, return_lse=True, deterministic=True)
    out_aiter.backward(dO_aiter)
    q_grad_aiter_bnhd = Q_aiter.grad
    k_grad_aiter_bnhd = K_aiter.grad  
    v_grad_aiter_bnhd = V_aiter.grad
    out_aiter_bhnd = out_aiter.transpose(1, 2)  # BNHD -> BHND
    q_grad_aiter_bhnd = q_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    k_grad_aiter_bhnd = k_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    v_grad_aiter_bhnd = v_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND

# PyTorch Reference
print("Running PyTorch reference...")
out_pytorch, q_grad_pytorch, k_grad_pytorch, v_grad_pytorch = reference_backwards(Q_pytorch, K_pytorch, V_pytorch, dO_pytorch, causal)

# Tiled Reference
print("Running Tiled forward to get m, l...")
QK = torch.matmul(Q_tiled.float(), K_tiled.transpose(-2, -1).float()) / math.sqrt(d)
m_tiled = QK.max(dim=-1, keepdim=True)[0] 
exp_scores = torch.exp(QK - m_tiled)  
l_tiled = exp_scores.sum(dim=-1, keepdim=True)  
P_tiled = exp_scores / l_tiled
O_tiled = torch.matmul(P_tiled, V_tiled.float())
m_tiled = m_tiled.squeeze(-1)
l_tiled = l_tiled.squeeze(-1)

dQ_tiled, dK_tiled, dV_tiled = simple_flash_backward(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), m_tiled, l_tiled)
out_tiled_bhnd = O_tiled
q_grad_tiled_bhnd = dQ_tiled
k_grad_tiled_bhnd = dK_tiled
v_grad_tiled_bhnd = dV_tiled


def test_dq(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S - torch.ones_like(m.unsqueeze(-1))) / torch.ones_like(l.unsqueeze(-1))
    O = torch.matmul(P, V)
    # # softmax backward
    Delta = torch.ones_like((dO * O).sum(dim=-1, keepdim=True))                 #
    dS = P * (torch.matmul(dO, V.transpose(-2, -1)) - Delta)   # (B, N, H, N)
    # chain rule through S = (Q K^T) * scale
    dQ = torch.matmul(dS, K) * scale
    return dQ


# TK
print("Running TK dQ ...")
dQ_tk = torch.zeros_like(q_grad_tiled_bhnd).bfloat16()
O_tk = O_tiled.bfloat16().clone()
dO_tk = dO_tiled.bfloat16().clone()
tk_kernel.dispatch_micro(
    Q_tk,     # Qg
    K_tk,     # Kg
    V_tk,     # Vg
    O_tk,     # Og
    dO_tk,    # dOg
    dQ_tk,    # dQg
    m_tiled.unsqueeze(-1),  # m_vec
    l_tiled.unsqueeze(-1)
)


def test_dv(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S - torch.ones_like(m.unsqueeze(-1))) / torch.ones_like(l.unsqueeze(-1))
    O = torch.matmul(P, V)
    # dV
    dV = torch.matmul(P.transpose(-2, -1), dO)
    return dV


def test_dk(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S - torch.ones_like(m.unsqueeze(-1))) / torch.ones_like(l.unsqueeze(-1))
    O = torch.matmul(P, V)
    # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)                 # (B, N, H, 1)
    dS = P * (torch.matmul(dO, V.transpose(-2, -1)) - torch.ones_like(Delta))   # (B, N, H, N)
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale
    return dK


DV = test_dk(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), m_tiled, l_tiled)

print("Running TK dK, dV ...")
dK_tk = torch.zeros_like(k_grad_tiled_bhnd).bfloat16()
dV_tk = torch.zeros_like(v_grad_tiled_bhnd).bfloat16()
test = torch.zeros_like(DV).bfloat16()
tk_kernel.dispatch_bwd_dkv(
    Q_tk,     # Qg
    K_tk,     # Kg
    V_tk,     # Vg
    O_tk,     # Og
    dO_tk,    # dOg
    dK_tk,    # dKg (output)
    dV_tk,    # dVg (output)
    m_tiled.unsqueeze(-1),  # m_vec
    l_tiled.unsqueeze(-1),  # l_vec
    test
)

diff = (DV - test).abs()
max_diff = diff.max()
if test.shape[-1] == 1:
    print(f"dV_tk[0, 0, :16]={test[0, 0, :16]}")
    print(f"DV[0, 0, :16]={DV[0, 0, :16]}")
else:
    print(f"dV_tk[0, 0, :16]={test[0, 0, 0, :16]}")
    print(f"DV[0, 0, :16]={DV[0, 0, 0, :16]}")
print(f"Max diff: {max_diff.item():.6f}")
# breakpoint()
exit()








# # Compare
# if use_aiter:
#     out_diff = (out_aiter_bhnd - out_pytorch).abs()
#     q_grad_diff = (q_grad_aiter_bhnd - q_grad_pytorch).abs()
#     k_grad_diff = (k_grad_aiter_bhnd - k_grad_pytorch).abs()
#     v_grad_diff = (v_grad_aiter_bhnd - v_grad_pytorch).abs()

# # Compare TK with PyTorch
# out_tiled_diff = (out_tiled_bhnd - out_pytorch).abs()
# q_grad_tiled_diff = (q_grad_tiled_bhnd - q_grad_pytorch).abs()
# k_grad_tiled_diff = (k_grad_tiled_bhnd - k_grad_pytorch).abs()
# v_grad_tiled_diff = (v_grad_tiled_bhnd - v_grad_pytorch).abs()

# if use_aiter:
#     print(f"\nOutput comparison:")
#     print(f"Output max error: {out_diff.max().item():.6f}")
#     print(f"Output mean error: {out_diff.mean().item():.6f}")

#     print(f"\nGradient comparison:")
#     print(f"Q grad max error: {q_grad_diff.max().item():.6f}")
#     print(f"K grad max error: {k_grad_diff.max().item():.6f}")
#     print(f"V grad max error: {v_grad_diff.max().item():.6f}")

# print(f"\nTiled vs PyTorch comparison:")
# print(f"Output max error: {out_tiled_diff.max().item():.6f}")
# print(f"Q grad max error: {q_grad_tiled_diff.max().item():.6f}")
# print(f"K grad max error: {k_grad_tiled_diff.max().item():.6f}")
# print(f"V grad max error: {v_grad_tiled_diff.max().item():.6f}")

# # TK vs PyTorch
# print(f"\nTK vs PyTorch comparison:")
# q_grad_tk_diff = (dQ_tk - q_grad_pytorch).abs()
# k_grad_tk_diff = (dK_tk - k_grad_pytorch).abs()
# v_grad_tk_diff = (dV_tk - v_grad_pytorch).abs()
# print(f"Q grad max error: {q_grad_tk_diff.max().item():.6f}")
# print(f"K grad max error: {k_grad_tk_diff.max().item():.6f}")
# print(f"V grad max error: {v_grad_tk_diff.max().item():.6f}")


# print(dQ_tk[0, 0, 0, :16])
# print(q_grad_pytorch[0, 0, 0, :16], q_grad_pytorch.max())
# if use_aiter:
#     print(q_grad_aiter_bnhd[0, 0, 0, :16], q_grad_aiter_bnhd.max())



