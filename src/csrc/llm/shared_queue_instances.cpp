// shared_queue_instances.cpp

#include "shared_queue.h"
#include "shared_queue_wrapper.h"

// Define the global queue instances here — single definitions
SharedQueueWrapper g_tensor_layer_queue;
SharedQueueWrapper g_kv_cache_queue;
SharedTimingInfoWrapper g_timing_info_var;