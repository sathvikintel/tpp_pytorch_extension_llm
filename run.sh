 ./run_llm.sh LLAMA-3.1 8B 1024 31744 4 1 50 malloc 0 0 0 temp.log 32 0 0 > run_1.log 2>&1

 sleep 10

 ./run_llm.sh LLAMA-3.1 8B 1024 31744 4 1 50 malloc 0 0 0 temp.log 32 0 1 > run_2.log 2>&1
