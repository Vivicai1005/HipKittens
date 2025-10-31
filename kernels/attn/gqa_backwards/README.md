# N = 1024
cd $THUNDERKITTENS_ROOT
git checkout asm_port
cd $THUNDERKITTENS_ROOT/../kernels/TK/attn/gqa_backwards
make SRC=attn_bkwd_non_causal.cpp TARGET=tk_kernel_bkwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024

cd $THUNDERKITTENS_ROOT
git checkout port
cd $THUNDERKITTENS_ROOT/../kernels/TK/attn/gqa_backwards
make SRC=attn_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024
make SRC=attn_fwd_non_causal.cpp TARGET=tk_kernel_fwd ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=1024

python test_python.py 16 1024 64 8 0 mi355x_gqa_bkwd_non_causal.json