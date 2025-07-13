if [ "$#" -ne 15 ]; then
    echo "Usage: $0 <model> <params> <inp_seq_len> <new_tokens> <batch_size> <iter> <threads> <allocator> <huge_pages> (0/1) <spill to hbm first?> (0/1) <tier (0/1) <hbm_dram_log> <N_L_HBM> <N_L_mig> <kv_static_pin>"
    exit 1
fi


model="$1"
params="$2"
inp_seq_len="$3"
new_tokens="$4"
batch_size="$5"
iter="$6"
threads="$7"
allocator="$8"
huge_pages="$9"
hbm_first="${10}"
tier="${11}"
hbm_dram_log="${12}"
n_l_hbm="${13}"
n_l_mig="${14}"
kv_static_pin="${15}"


export LD_PRELOAD=/data/sathvik/tpp-pytorch-extension/miniforge3/envs/pt251/lib/libcrypt.so.2:/data/sathvik/tpp-pytorch-extension/miniforge3/envs/pt251/lib/libpython3.9.so.1.0:/data/sathvik/tpp-pytorch-extension/miniforge3/envs/pt251/lib/libtcmalloc.so:/data/sathvik/tpp-pytorch-extension/miniforge3/envs/pt251/lib/libiomp5.so


# if [ "$#" -ne 0]; then
#     echo "Usage: $0"
#     exit 1
# fi

# source "run_llm.cfg"


echo 3 > /proc/sys/vm/drop_caches


# Turn on tracing 
echo 1 > /sys/kernel/debug/tracing/tracing_on

echo 1 > /sys/kernel/debug/tracing/events/migrate/migrate_folio_unmap/enable
echo 1 > /sys/kernel/debug/tracing/events/migrate/migrate_pages_batch/enable
echo 1 > /sys/kernel/debug/tracing/events/migrate/kernel_move_pages/enable
echo 1 > /sys/kernel/debug/tracing/events/migrate/do_move_pages_to_node/enable
echo 1 > /sys/kernel/debug/tracing/events/migrate/move_pages/enable

echo x86-tsc > /sys/kernel/debug/tracing/trace_clock

ALLOCATOR=$allocator
NUM_THREADS=$threads
MIGRATE_THREADS=1


pkill -9 pcm-memory

kill_tpp_python_process() {
    # Extract the PID of the target process
    pid=$(ps aux | grep 'python tpp-pytorch-extension' | grep -v 'grep' | awk '{print $2}')
    
    # Check if the PID exists
    if [[ -n "$pid" ]]; then
        echo "Killing process with PID: $pid"
        kill -9 "$pid"
        pkill -9 python 
        echo "Process $pid killed successfully."
    fi
}

if [ "$huge_pages" -eq 0 ]; then
    echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
    pages="base_pages"
elif [ "$huge_pages" -eq 1 ]; then
    echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
    pages="huge_pages"
else
    echo "Invalid value for huge_pages. It should be 0 or 1."
fi

pkill -9 python
pkill -9 perf
pkill -9 pcm-memory

sleep 5 

time=$(date +"%Y%m%d-%H%M%S")

TOP_LVL_DIR="eval_/${model}_${params}/${NUM_THREADS}_threads/${ALLOCATOR}/${pages}/inp_len_${inp_seq_len}/out_len_${new_tokens}/batch_size_${batch_size}/iter_${iter}/${time}"
EXEC_DIR="${TOP_LVL_DIR}/exec"
PLOTS_DIR="${TOP_LVL_DIR}/plots"
PERF_STAT_LOG_DIR="/home/sathvik/LLM/PERF_STATS"
NAME=LLM_${model}_${params}_${NUM_THREADS}_threads_${ALLOCATOR}_${pages}_inp_len_${inp_seq_len}_out_len_${new_tokens}_batch_size_${batch_size}_iter_${iter}

TREND_DIR=/home/sathvik/scripts
PERF_EVENTS=$(cat ${TREND_DIR}/perf-all-fmt)
CONT_PERF_EVENTS=$(cat ${TREND_DIR}/perf-trend-fmt)
PERF_TIMER=1000

bw_log_file=${EXEC_DIR}/mem_bw_log.txt
rss_csv_file=${EXEC_DIR}/rss.csv
summary_file=${EXEC_DIR}/summary.log
rss_output_file=${PLOTS_DIR}/rss_plot.png
llm_perf_stat_file=$PERF_STAT_LOG_DIR/${NAME}_perf_stat.log
llm_load_model_perf_stat_file=${EXEC_DIR}/load_model_perf_stat.log
llm_first_token_perf_stat_file=${EXEC_DIR}/first_token_perf_stat.log
llm_second_token_plus_perf_stat_file=${EXEC_DIR}/second_token_plus_perf_stat.log
numa_mem_file=${PLOTS_DIR}/node_wise_mem.png
mem_pattern_mmap_output_file=${PLOTS_DIR}/mmap_mem_pattern.png
mem_pattern_brk_output_file=${PLOTS_DIR}/brk_mem_pattern.png
mem_pattern_output_file=${PLOTS_DIR}/mem_pattern.png
strace_log_file=${EXEC_DIR}/strace.log
brk_alloc_plot_file=${PLOTS_DIR}/brk_alloc.png
kv_cache_mem_pattern_output_file=${PLOTS_DIR}/kv_cache_mem_pattern.png
weights_mem_pattern_output_file=${PLOTS_DIR}/weights_mem_pattern.png
activation_cache_mem_pattern_output_file=${PLOTS_DIR}/activation_cache_pattern.png
llm_perf_stat_ratio_file=${PLOTS_DIR}/llm_perf_stat_ratio.png
perf_lock_time=${EXEC_DIR}/perf_lock.data
user_kernel_time=${EXEC_DIR}/user_kernel_time.log
lock_trace_file=${EXEC_DIR}/lock_trace_file.log
pmap_output_file=${EXEC_DIR}/pmap_output.log
user_kernel_cycles_plot=${PLOTS_DIR}/user_kernel_cycles_plot.png
DRAM_read_mem_bw_output_file=${PLOTS_DIR}/DRAM_RD_BW.png
DRAM_write_mem_bw_output_file=${PLOTS_DIR}/DRAM_WR_BW.png
HBM_read_mem_bw_output_file=${PLOTS_DIR}/HBM_RD_BW.png
HBM_write_mem_bw_output_file=${PLOTS_DIR}/HBM_WR_BW.png
user_kernel_time_overall=${EXEC_DIR}/user_kernel_cycles.log
mhsr_log_file=${EXEC_DIR}/mshr_counters.log 
mshr_plot_file=${PLOTS_DIR}/mshr_counters_plot.png 
mshr_overall_log_file=${EXEC_DIR}/mshr_counters_total.log
memory_access_cycles_log_file=${EXEC_DIR}/mem_access_cycles.log
cache_misses_log=${EXEC_DIR}/cache_misses.log
cache_misses_plot=${PLOTS_DIR}/cache_misses.png
l2mpki_log=${EXEC_DIR}/l2mpki.log
l2mpki_plot=${PLOTS_DIR}/l2mpki.png
stalls_log=${EXEC_DIR}/stalls.log
stalls_plot=${PLOTS_DIR}/stalls.png
tma_log=${EXEC_DIR}/tma_stats.log
tma_log_per_core=${EXEC_DIR}/tma_stats_per_core.log
tma_mem_log=${EXEC_DIR}/tma_mem_bound_stats.log
tma_core_log=${EXEC_DIR}/tma_core_bound_stats.log
tma_serizalizing_core_log=${EXEC_DIR}/tma_serializing_core_bound_stats.log
perf_lock_file=${EXEC_DIR}/perf_lock_log.txt
tma_pause_cycles_log=${EXEC_DIR}/tma_pause_cycles.log
tma_pause_cycles_plot=${PLOTS_DIR}/tma_pause_cycle.png 
k_cache_mem_pattern_output_file=${PLOTS_DIR}/K_vector_layer0.png 
v_cache_mem_pattern_output_file=${PLOTS_DIR}/V_vector_layer0.png 
c_cache_mem_pattern_output_file=${PLOTS_DIR}/C_vector_layer0.png 
d_cache_mem_pattern_output_file=${PLOTS_DIR}/D_vector_layer0.png 
tma_pause_cycles_per_core_plot=${PLOTS_DIR}/pause_cycles_per_core.png
llm_token_time_split=${PLOTS_DIR}/llm_token_time_split.png
tma_pause_cycles_log_total=${EXEC_DIR}/tma_pause_cycles_log_total.log
ipc_log=${EXEC_DIR}/ipc.log 
tiering_daemon_perf_stat_file=${EXEC_DIR}/tiering_daemon_perf_stats.log
move_pages_to_dram_split_log=${EXEC_DIR}/move_pages_to_dram_split.log
move_pages_to_dram_split_image=${PLOTS_DIR}/move_pages_to_dram_split.png
move_pages_to_hbm_split_log=${EXEC_DIR}/move_pages_to_hbm_split.log
move_pages_to_hbm_split_image=${PLOTS_DIR}/move_pages_to_hbm_split.png
kv_addr_size_log=${EXEC_DIR}/kv_cache_size.log
attn_kernel_latency_log=${EXEC_DIR}/attn_kernel_latency.log
ffn_kernel_latency_log=${EXEC_DIR}/ffn_kernel_latency.log
ffn_attn_plot=${PLOTS_DIR}/ffn_attn_latency.png
ffn_mha_perf_metrics_log=${EXEC_DIR}/ffn_mha_perf_metrics.log

mkdir -p ${EXEC_DIR}
mkdir -p ${PLOTS_DIR}
mkdir log_files/

MSHR_COUNTERS="l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,offcore_requests_outstanding.data_rd,offcore_requests_outstanding.cycles_with_data_rd,offcore_requests_outstanding.demand_data_rd,offcore_requests_outstanding.cycles_with_demand_data_rd,offcore_requests.data_rd,memory_activity.stalls_l1d_miss,memory_activity.stalls_l2_miss"

touch $bw_log_file
initial_time=$(date +%s)
output_file="log_files/utc_start_time.txt"
touch "scripts/rss_log"
> scripts/rss_log

> $output_file
> log_files/log_lines.txt
> log_files/llm_events.log 
> log_files/kv_layer_addr.log
> log_files/call_stack.log
> log_files/llm_perf_stats.log 
> log_files/llm_gemm_phase_mem_usage.log
> log_files/token_layer_timestamps.log
> tier_infer/log_files/llm_mem_migrate_daemon_status.log
> tier_infer/log_files/llm_mem_region_migrate.log
> kv_cache_size.log
> log_files/attn_kernel_latency.log
> llm_mem_region_migrate.log 
> log_files/ffn_kernel_latency.log
> log_files/ffn_mha_perf_metrics.log

# Check and set the LD_PRELOAD based on the allocator variable
if [ "$ALLOCATOR" == "tcmalloc" ]; then
    if [ -f /usr/lib/x86_64-linux-gnu/libtcmalloc.so ]; then
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so:$LD_PRELOAD
        echo "TCMalloc is set as the memory allocator." > $summary_file
    else
        echo "TCMalloc library not found at /usr/lib/x86_64-linux-gnu/libtcmalloc.so"  > $summary_file
    fi
elif [ "$ALLOCATOR" == "jemalloc" ]; then
    if [ -f /usr/lib64/libjemalloc.so.2 ]; then
        export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD
        echo "Jemalloc is set as the memory allocator." > $summary_file
    else
        echo "Jemalloc library not found at /usr/lib64/libjemalloc.so.2." > $summary_file
    fi
else
    export LD_PRELOAD=$LD_PRELOAD
    echo "System malloc is set as the memory allocator." > $summary_file
fi

# export LD_PRELOAD=/home/sathvik/tpp-pytorch-extension/miniforge3/lib/libcrypt.so.2:$LD_PRELOAD
# export LD_PRELOAD=:/home/sathvik/tpp-pytorch-extension/miniforge3/lib/libpython3.12.so.1.0:$LD_PRELOAD
# export LD_LIBRARY_PATH=/home/sathvik/tpp-pytorch-extension/miniforge3/:$LD_LIBRARY_PATH
echo $PROFILE_TOKEN
echo $PROFILE_LAYER

echo "$initial_time" > "$output_file"

touch $summary_file

echo "Summary file: $summary_file"

cpu_count=$(numactl -H | grep "cpus:" | awk -F': ' '{print $2}' | tr ' ' '\n' | grep -v '^$' | wc -l)

smt_file="/sys/devices/system/cpu/smt/control"

# Check if the file exists
if [ ! -f "$smt_file" ]; then
    echo "SMT control file not found: $smt_file"
    exit 1
fi

# Extract the active state by grabbing the text inside brackets
smt_status=$(cat $smt_file)

echo "CPU count is $cpu_count"
echo "SMT is $smt_status"

single_socket=0

if [ "$smt_status" = "off" ] && [ "$cpu_count" -le 56 ]; then
    single_socket=1
    echo "Running in a single socket"
elif [ "$smt_status" = "on" ] && [ "$cpu_count" -le 112 ]; then
    single_socket=1
    echo "Running in a single socket"
fi


# ----------------------------------------------------------------------------
#                       RUN LLM WORKLOAD
#-----------------------------------------------------------------------------

echo 0 > /proc/sys/kernel/nmi_watchdog  


if [ $model = "GPTJ" ] ; then
OMP_NUM_THREADS=$NUM_THREADS KV_CACHE_INC_SIZE=4096 /home/sathvik/numactl/numactl  -N 0,1,2,3,4,5 --localalloc python -u examples/llm/run_generation.py --max-new-tokens $new_tokens --device cpu --dtype bfloat16 --input-tokens $inp_seq_len --batch-size $batch_size --use-tpp --num-iter $iter --num-warmup 0 --token-latency -m EleutherAI/gpt-j-6b  --summary_file $summary_file &
fi
if [ $model = "LLAMA-2" ] ; then
OMP_NUM_THREADS=$NUM_THREADS KV_CACHE_INC_SIZE=4096 /home/sathvik/numactl/numactl -m 0 -N 0,1 python -u examples/llm/run_generation.py -m meta-llama/Llama-2-7b-hf --use-tpp --token --batch-size $batch_size --dist-backend ccl --max $new_tokens --input $inp_seq_len --greedy --num-warmup 0 --num-iter $iter &
fi

if [ $model = "LLAMA-3" ] ; then
# 8B
OMP_NUM_THREADS=$NUM_THREADS KV_CACHE_INC_SIZE=$new_tokens+$inp_seq_len /home/sathvik/numactl/numactl -N 0 -m 0 python -u examples/llm/run_generation.py -m meta-llama/Meta-Llama-3-8B-Instruct --use-tpp --token --batch-size $batch_size --dist-backend ccl --max $new_tokens --input $inp_seq_len --greedy --num-warmup 0 --num-iter $iter --summary-file $summary_file &
# 70B
# OMP_NUM_THREADS=$NUM_THREADS KV_CACHE_INC_SIZE=8192 /home/sathvik/numactl/numactl -N 0 -w 0,2 python -u examples/llm/run_generation.py -m meta-llama/Meta-Llama-3-70B --use-tpp --token --batch-size $batch_size --dist-backend ccl --max $new_tokens --input $inp_seq_len --greedy --num-warmup 0 --num-iter $iter --summary-file $summary_file &
fi
if [ $model = "LLAMA-3.1" ] ; then
# 8B
OMP_NUM_THREADS=$NUM_THREADS KV_CACHE_INC_SIZE=$new_tokens+$inp_seq_len /home/sathvik/numactl/numactl -m 0 -N 0 python -u examples/llm/run_generation.py -m meta-llama/Llama-3.1-8B --use-tpp --token --batch-size $batch_size --dist-backend ccl --max $new_tokens --input $inp_seq_len --greedy --num-warmup 0 --num-iter $iter --summary-file $summary_file &
# 70B
# OMP_NUM_THREADS=$NUM_THREADS KV_CACHE_INC_SIZE=8192 /home/sathvik/numactl/numactl -m 0 -N 0 python -u examples/llm/run_generation.py -m meta-llama/Meta-Llama-3-70B --use-tpp --token --batch-size $batch_size --dist-backend ccl --max $new_tokens --input $inp_seq_len --greedy --num-warmup 0 --num-iter $iter --summary-file $summary_file &
fi

python_script_pid=$!

echo "PID: $python_script_pid"

# ----------------------------------------------------------------------------
#                       MONITORING SCRIPTS
#-----------------------------------------------------------------------------

if [ "$tier" -eq 1 ]; then
    echo "Tier infer invoked"
    tier_infer/tier_infer $python_script_pid tier_infer/log_files/llm_mem_region_migrate.log tier_infer/log_files/llm_mem_migrate_daemon_status.log $MIGRATE_THREADS $n_l_mig $n_l_hbm &
    llm_tiering_daemon_pid=$!
else
    # You can add commands here for the else branch, or leave it empty
    :
fi

# if [ "$tier" -eq 1 ]; then
#     ./migrate_llm_memory_daemon $python_script_pid llm_mem_region_migrate.log llm_mem_migrate_daemon_status.log $MIGRATE_THREADS &
#     llm_tiering_daemon_pid=$!
# else
    # You can add commands here for the else branch, or leave it empty
    :
# fi

if [ "$kv_static_pin" -eq 1 ]; then
    pin_kv/pin_kv $python_script_pid kv_cache_size.log 0 &
    pin_kv_pid=$!
else
    # You can add commands here for the else branch, or leave it empty
    :
fi

# # monitor data sharing across threads
# ps -T -p $python_script_pid > $pmap_output_file &

# perf stat -e instructions,cycles -p $python_script_pid -o $ipc_log &
# #monitor generic PMU counters
# perf stat -x, -o $llm_perf_stat_file -e $PERF_EVENTS -p $python_script_pid &

# # perf stat -x, -o $tiering_daemon_perf_stat_file -e $PERF_EVENTS -p $llm_tiering_daemon_pid &

# perf stat -p $llm_tiering_daemon_pid &
# monitor mshr related counter
#perf stat -x , -o $mhsr_log_file -e $MSHR_COUNTERS -I $PERF_TIMER -p $python_script_pid & 

#perf stat -x , -o $mshr_overall_log_file -e $MSHR_COUNTERS  -p $python_script_pid & 

# record lock contention time
#perf lock record -p $python_script_pid --output=$perf_lock_time &

#perf stat -x, -o $llm_load_model_perf_stat_file -e $PERF_EVENTS -p $python_script_pid &
# load_model_perf=$!

# record user space and kernel time 
#perf stat -o $user_kernel_time -e cycles:u,cycles:k -I $PERF_TIMER -p $python_script_pid &

#perf stat -o $user_kernel_time_overall -e cycles:u,cycles:k -p $python_script_pid &

#perf stat -o $memory_access_cycles_log_file -e mem_inst_retired.all_loads,mem_inst_retired.all_stores,cycles -p $python_script_pid &

#perf stat -e mem_load_retired.l2_miss -I $PERF_TIMER -o $cache_misses_log -p $python_script_pid &

#perf stat -e cycle_activity.stalls_l1d_miss,cycle_activity.stalls_l2_miss,cycle_activity.stalls_l3_miss,cycle_activity.cycles_mem_any,cycle_activity.stalls_total,dtlb_load_misses.walk_active,dtlb_store_misses.walk_active,cycles,memory_activity.stalls_l3_miss -I $PERF_TIMER -o $stalls_log -p $python_script_pid &

# monitor node-wise memory consumption
scripts/monitor_numastat.sh $python_script_pid ${EXEC_DIR} &
numastat_pid=$!

# # monitor memory bandwidth 
# python scripts/run_pcm_until_pid.py $python_script_pid $bw_log_file &
# pcm_process_pid=$!

# monitor memory consumption
python scripts/measure_rss.py $python_script_pid 0 -o $rss_output_file --summary-file $summary_file & 

# monitor memory access patterns
#perf record -o ${EXEC_DIR}/mem_access_perf.data -d -e cpu/event=0xd0,umask=0x81/ppu -e cpu/event=0xd0,umask=0x82/ppu -e cpu/event=0xd0,umask=0x11/ppu -e cpu/event=0xd0,umask=0x12/ppu -p $python_script_pid &

# monitor CPU utilization
python scripts/monitor.py --pid $python_script_pid -o ${PLOTS_DIR}/cpu_util.png --avg_output ${PLOTS_DIR}/cpu_util_avg.png &

#Record call graph 
# perf record -o ${EXEC_DIR}/call_graph_perf.data -a -g -p $python_script_pid &

# # Record memory allocations
# strace -ttt -e trace=mmap,brk,munmap -o $strace_log_file -p $python_script_pid &

# Function to start /usr/bin/#perf stat
start_perf_stat() {
    echo "Starting /usr/bin/#perf stat..."
    #perf stat -x, -o $llm_first_token_perf_stat_file -e $PERF_EVENTS -p $python_script_pid &
    PERF_PID=$!  # Save the PID of the /usr/bin/#perf command
}

# Function to stop /usr/bin/#perf stat
stop_perf_stat() {
    echo "Stopping /usr/bin/#perf stat..."
    if [ -n "$PERF_PID" ]; then
        kill -SIGINT "$PERF_PID"  # Send interrupt signal to stop perf
        wait "$PERF_PID" 2>/dev/null  # Wait for the process to exit
        PERF_PID=""
    fi
}

stop_load_llm_perf_stat() {
    echo "Stopping /usr/bin/#perf stat..."
    if [ -n "$load_model_perf" ]; then
        kill -SIGINT "$load_model_perf"  # Send interrupt signal to stop perf
        wait "$load_model_perf" 2>/dev/null  # Wait for the process to exit
        load_model_perf=""
    fi
}

# Check if the PID exists, and read /proc/vmstat while the process is running
while kill -0 $python_script_pid 2>/dev/null; do
    sleep 1
done



kill -9 $numastat_pid &
kill -9 $pcm_process_pid &
kill -9 $llm_tiering_daemon_pid &
kill -9 $pin_kv_pid &
pkill pcm-memory 

wait 

echo 1 > /proc/sys/kernel/nmi_watchdog

cp log_files/ffn_mha_perf_metrics.log $ffn_mha_perf_metrics_log &

cp log_files/attn_kernel_latency.log $attn_kernel_latency_log &

cp log_files/ffn_kernel_latency.log $ffn_kernel_latency_log &

cp kv_cache_size.log $kv_addr_size_log &

cat /sys/kernel/debug/tracing/trace  | grep tier_infer | grep -e "target_node=1" -e "target_node=0" > $move_pages_to_dram_split_log &

cat /sys/kernel/debug/tracing/trace  | grep tier_infer | grep -e "target_node=2" -e "target_node=3" > $move_pages_to_hbm_split_log &

python scripts/format_prefill_perf_stats.py &

# cp /home/sathvik/tpp-pytorch-extension/llm_gemm_phase_mem_usage.log ${EXEC_DIR}/gemm_mem_usage.log &

# python scripts/plot_pause_cycles_chart.py $tma_pause_cycles_log $tma_pause_cycles_per_core_plot &

#perf lock report -i $perf_lock_time -F wait_total --output=$perf_lock_file &

python scripts/plot_user_kernel_cycles.py $user_kernel_time  $user_kernel_cycles_plot $summary_file &

python scripts/plot_mshr_counters.py $mhsr_log_file $mshr_plot_file &

python scripts/plot_memtd.py ${EXEC_DIR}/numastat_output.txt $numa_mem_file 0 $rss_csv_file $summary_file &

#perf script -i ${EXEC_DIR}/mem_access_perf.data -F trace: -F time,addr --reltime > ${EXEC_DIR}/perf_log.txt &

python scripts/smap_process.py -i $strace_log_file -tmp ${EXEC_DIR} &


wait 

python scripts/attn_latency.py $attn_kernel_latency_log ${PLOTS_DIR} &

python scripts/plot_ffn_attn_latency.py $ffn_kernel_latency_log $attn_kernel_latency_log $ffn_attn_plot &

python scripts/viz_move_pages_split.py $move_pages_to_dram_split_log $move_pages_to_dram_split_image $summary_file &

python scripts/viz_move_pages_split.py $move_pages_to_hbm_split_log $move_pages_to_hbm_split_image $summary_file &

python scripts/perf_func_wise.py --output-dir ${PLOTS_DIR} --summary-file $summary_file &

# python scripts/plot_tma_per_core.py $tma_log_per_core ${PLOTS_DIR} &

python scripts/compute_lock_time.py $perf_lock_file $summary_file &

python scripts/plot_cache_misses.py $cache_misses_log $cache_misses_plot &

python scripts/plot_stalls.py $stalls_log $stalls_plot $summary_file &


python scripts/plot_l2mpki.py $l2mpki_log $l2mpki_plot &

python scripts/plot_mem_bw.py $bw_log_file $DRAM_read_mem_bw_output_file 0 0 $single_socket  $summary_file &

python scripts/plot_mem_bw.py $bw_log_file $DRAM_write_mem_bw_output_file 0 1 $single_socket  $summary_file &

python scripts/plot_mem_bw.py $bw_log_file $HBM_read_mem_bw_output_file 0 2 $single_socket  $summary_file &

python scripts/plot_mem_bw.py $bw_log_file $HBM_write_mem_bw_output_file 0 3 $single_socket  $summary_file &

python scripts/perf_to_pebs.py ${EXEC_DIR}/perf_log.txt ${EXEC_DIR}/pebs_log.txt &

python scripts/viz_llm_perf_counters.py $llm_load_model_perf_stat_file $llm_first_token_perf_stat_file $llm_second_token_plus_perf_stat_file $llm_perf_stat_ratio_file &

python scripts/plot_mmap.py -o $PLOTS_DIR -tmp ${EXEC_DIR} &

python scripts/brk_plot.py ${EXEC_DIR}/brk_strace_log.txt $brk_alloc_plot_file &

kv_layer_cache_size=$(awk '/Layer 32:/ {f=1} f && /Total KV cache size:/ {print $5; exit}' $kv_addr_size_log)

kv_total=$(echo "$kv_cache_layer_size * 32" | bc -l)
echo "KV cache size: $kv_total GB" >> "$summary_file"


# python scripts/scatter_based_pebs.py -i ${EXEC_DIR}/pebs_log.txt -o $mem_pattern_output_file -m 5 &

# python scripts/scatter_based_pebs.py -i ${EXEC_DIR}/pebs_log.txt -o $kv_cache_mem_pattern_output_file -min $kv_min_addr -max $kv_max_addr -m 5 &

# python scripts/scatter_based_pebs.py -i ${EXEC_DIR}/pebs_log.txt -o $weights_mem_pattern_output_file -min $weights_min_addr -max $weights_max_addr -m 5 &

# python scripts/scatter_based_pebs.py -i ${EXEC_DIR}/pebs_log.txt -o $activation_cache_mem_pattern_output_file -min $activation_min_addr -max $activation_max_addr -m 5 &

# python scripts/scatter_based_pebs.py -i ${EXEC_DIR}/pebs_log.txt -o $k_cache_mem_pattern_output_file -min $k_min_addr -max $k_max_addr -m 15 &

# python scripts/scatter_based_pebs.py -i ${EXEC_DIR}/pebs_log.txt -o $v_cache_mem_pattern_output_file -min $v_min_addr -max $v_max_addr -m 15 &

# python scripts/scatter_based_pebs.py -i ${EXEC_DIR}/pebs_log.txt -o $c_cache_mem_pattern_output_file -min $act_c_min_addr -max $act_c_max_addr -m 15 &

# python scripts/scatter_based_pebs.py -i ${EXEC_DIR}/pebs_log.txt -o $d_cache_mem_pattern_output_file -min $act_d_min_addr -max $act_d_max_addr  -m 15 &

# python scripts/plot_pie_chart.py llm_token_time_split.log $llm_token_time_split &

wait 

cp $llm_perf_stat_file ${EXEC_DIR}/
echo "Check $summary_file"
echo "Check $move_pages_to_dram_split_log"
echo "Check $move_pages_to_dram_split_image"
echo "Check $move_pages_to_hbm_split_log"
echo "Check $move_pages_to_hbm_split_image"

# echo "Check $bw_log_file"

# echo "Number of threads: $threads" >> $hbm_dram_log
# echo "Mode: $hbm_first (0: FM_DRAM / 1: FM_HBM)" >> $hbm_dram_log
# echo $(cat $summary_file| grep "Inference latency") >> $hbm_dram_log
# echo $(cat $summary_file| grep "First token average latency:") >> $hbm_dram_log
# echo $(cat $summary_file| grep "Average 2... latency:") >> $hbm_dram_log
# echo "------------------------" >> $hbm_dram_log
# echo "Check $hbm_dram_log"
# echo "Check $perf_lock_time"
echo "Check ${EXEC_DIR}/call_graph_perf.data"
# echo "Check $user_kernel_time"
# echo "Check $lock_trace_file"
# echo "Check $pmap_output_file"
# echo "Check $user_kernel_cycles_plot"
# echo "Check $user_kernel_time_overall"
# echo "Check $mshr_plot_file"
# echo "Check $mshr_overall_log_file"
# echo "Check $memory_access_cycles_log_file"
# echo "Check $cache_misses_plot"
# echo "Check $l2mpki_log"
# echo "Check $llm_perf_stat_file"
# echo "Check $l2mpki_plot"
# echo "Check $stalls_log"
# echo "Check $ipc_log"
# echo "Check $llm_perf_stat_file"
# echo "Check $tiering_daemon_perf_stat_file"
echo 3 > /proc/sys/vm/drop_caches
echo "Check $kv_addr_size_log"
echo "Check $ffn_attn_plot"
echo "Check $ffn_mha_perf_metrics_log"