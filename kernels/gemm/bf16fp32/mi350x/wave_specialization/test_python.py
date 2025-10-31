import torch
import tk_kernel
import random
import time
from aiter.tuned_gemm import tgemm

profiling = True
torch.manual_seed(0)
random.seed(0)

# Inputs
N = 32
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
Bt = B.t().contiguous()  # Transpose B for the kernel

if profiling:
    num_warmup = 1
    num_iters = 1
else:
    num_warmup = 1
    num_iters = 0

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = (2 * N**3)  # FLOPs for reference

for _ in range(num_warmup):
    C_ref = torch.matmul(A, Bt)
timings_ref = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    C_ref = torch.matmul(A, Bt)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_ref.append(elapsed_time)
if profiling:
    print(f"{C_ref.dtype=}")
    avg_time_ref = sum(timings_ref) / len(timings_ref)
    tflops_ref = flops_ref / (avg_time_ref * 1e9) 
    print(f"PyTorch reference average execution time: {avg_time_ref:.4f} ms")
    print(f"PyTorch reference performance: {tflops_ref:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


for _ in range(num_warmup):
    C_aiter = tgemm.mm(A, B, None, None, None)
timings_aiter = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    C_aiter = tgemm.mm(A, B, None, None, None)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_aiter.append(elapsed_time)
if profiling:
    print(f"{C_aiter.dtype=}")
    avg_time_aiter = sum(timings_aiter) / len(timings_aiter)
    tflops_aiter = flops_ref / (avg_time_aiter * 1e9) 
    print(f"AITER (AMD) reference average execution time: {avg_time_aiter:.4f} ms")
    print(f"AITER (AMD) reference performance: {tflops_aiter:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


# Kernel matmul
C = torch.zeros(N, N, dtype=torch.float, device='cuda')
for _ in range(num_warmup):
    tk_kernel.dispatch_micro(A, B, C)
timings = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_micro(A, B, C)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
if profiling:
    print(f"{C.dtype=}")
    avg_time = sum(timings) / len(timings)
    tflops = flops_ref / (avg_time * 1e9) 
    print(f"Average execution time: {avg_time:.4f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")

# Compare against reference
if profiling:
    C_float = C.float()
    C_ref_float = C_ref.float()
    diff = (C_float - C_ref_float).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    error_count = (diff > 0.1).sum().item()

    print(f"Max error between kernel and reference: {max_error}")
    print(f"Max error: {max_error}")
    print(f"Mean error: {mean_error}")
    print(f"Number of large errors (>0.1): {error_count}\n")

    # pos_max_diff = diff.max()
    # pos_max_diff_index = torch.where(diff == pos_max_diff)

    num_print = 8
    print("C Kernel:")
    print(C[:num_print, :num_print])
    print("C Reference:")
    print(C_ref[:num_print, :num_print])
    print("Diff:")
    print(diff[:num_print, :num_print])



