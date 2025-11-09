# Intel TPP with TierInfer Setup and Usage Guide

This guide walks through the steps to set up Intel TPP with TierInfer and run the workload.

---

## Cloning Repository with Submodules

Clone the repository including submodules:

```
git clone --recurse-submodules https://github.com/sathvikintel/tpp_pytorch_extension_llm.git
```

---

## Setup Environment

1. Change directory:

```
cd tpp_pytorch_extension_llm
```

2. Setup Conda environment:

```
bash utils/setup_conda.sh
source env.sh
```
---

## Install Torch CCL (for distributed node compute)

1. Navigate to utils directory:

```
cd utils/
```

2. Install Torch Collective Communications Library (CCL):

```
bash install_torch_ccl.sh
```

Note: Ensure your cmake version in `torch-ccl/third_party/oneCCL/CMakeLists.txt` is at least **3.5**.

3. Return to root directory:

```
cd ..
```

---

## Install Python Package and Dependencies

1. Install the Python package:

```
python setup.py install
```

2. Install example dependencies:

```
cd examples/llm/
pip install -r requirements.txt
cd ../../
```

---

## Build TierInfer

1. Change directory to tier_infer:

```
cd tier_infer/
```

2. Build with make:

```
make
```

3. Return to root directory:

```
cd ..
```

---

## Rebuild Intel TPP

1. Uncomment `tier_infer/lib_tier_llm_dynamic_partition.so` in `setup.py`

2. Reuild Intel TPP

```
python setup.py install
```
 
---

## Update LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=$LD_PRELOAD:$(pwd)/tier_infer/
```

---

## TPP Env vars

Control number of thread using  `OMP_NUM_THREADS` and KV cache allocation granularity using `KV_CACHE_INC_SIZE`

---

## Run Workload

Run the example workload:

```
python -u examples/llm/run_generation.py -m meta-llama/Meta-Llama-3-70B --use-tpp --token --batch-size 1 --dist-backend ccl --max 32 --input 128 --greedy --num-warmup 0 --num-iter 1 --summary-file summary.log
```

---

## Communicating tensors from TPP to TierInfer

Go to `src/csrc/llm/fused_llm_infer.cpp`

Pass relevant tensor to `send_to_tier_llm()`

---

## Disabling TierInfer

To disable TierInfer profiling/inference threads, comment out the following lines in `examples/llm/run_generation.py`:

```python 
dyn_thread.start()
inter_thread.start()
```

---
