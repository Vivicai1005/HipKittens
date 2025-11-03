# HipKittens

HipKittens is a repository in the ThunderKittens cinematic universe! This work provides minimal, opinionated C++ embedded programming primitives to help you write speedy AMD AI kernels. HipKittens is built from the hardware up -- we do what the silicon tells us. We support CDNA3 and CDNA 4. 

HK uses:
1. Tile primitives: sized according to the tensor core units. Tile memory ops are coalesced, bank conflict free, and eagerly use tensor core layouts. We focus on minimizing address computation costs. 
2. Python-inspired functions: bulk compute functions that operate over tiles. These are lightweight, wrapping assembly and HIP.
3. Asynchronous loads/stores: hide latencies and address generation using direct buffer loads to shared memory.
4. Scheduling and overlapping: we show two core patterns for overlapping compute and memory -- 8-wave ping pong and 4-wave interelave -- that appear across kernels.


<div align="center" >
    <img src="assets/hipkittens.png" height=250 alt="HipKittens logo" style="margin-bottom:px"/> 
</div>

<br>
<br>

## Setup

```bash
# clone the repo
git clone git@github.com:HazyResearch/HipKittens.git

# obtain an amd docker using docker pull or podman pull
podman pull docker.io/rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta

# enter the docker
podman run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir/ \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
    bash

# set the environment variables
cd HipKittens/
source env.src
```

## Unit tests

We provide unit tests for you to optionally test the correctness of library functions. 

```bash
cd HipKittens/tests/unit
make -j64
```

### Run kernels

GEMM:
```bash
# gemm kernel
cd kernels/TK/gemm/bf16fp32/mi325x/256_256_64_16/
make clean && make
python test_python.py
```

### Resources

We provide more docker information in:
```bash
docs/launch_docker_mi300x.md
docs/launch_docker_mi350x.md
```

We provide information on how to setup the profiler in:
```bash
docs/profiling_instructions.md
```

We provide a bash script to capture pm counters and a python script to view the outputs:
```bash
bash profile.sh
python analyze_prof.py # you will need to change the kernel names in here based on the kernels you're profiling
```

We provide a script to extract the assembly from rocprof generated output json files. 
```bash
extract_assembly.py [input.json] [output.s]
```

Contribute to our [onboarding documents](https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing).


### Get in touch!

Contact: William Hu [willhu@stanford.edu](willhu@stanford.edu) and Simran Arora [simran@cs.stanford.edu](simran@cs.stanford.edu).
Join us on Discord to get involved, [invitation link](https://discord.com/channels/1189498204333543425/1300872762163728550)!


