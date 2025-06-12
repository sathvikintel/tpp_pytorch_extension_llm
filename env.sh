#!/bin/bash

source /data/sathvik/tpp-pytorch-extension/miniforge3/bin/activate pt251
torch_ccl_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))" 2> /dev/null | grep oneccl_bind_pt |tail -n 1)
if test -f $torch_ccl_path/env/setvars.sh ; then
  source $torch_ccl_path/env/setvars.sh
fi

NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk '{print $NF}')
ARCH=$(lscpu | grep "Architecture" | awk '{print $NF}')
export OMP_NUM_THREADS=${NUM_THREADS}
if [ $(uname -m) == "x86_64" ] ; then
  export KMP_AFFINITY=compact,1,granularity=fine
  export KMP_BLOCKTIME=1
  export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so
fi
