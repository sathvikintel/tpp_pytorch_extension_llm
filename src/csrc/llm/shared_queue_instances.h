// shared_queue_instances.h
#include "shared_queue_wrapper.h"
#ifndef SHARED_QUEUE_INSTANCES_H
#define SHARED_QUEUE_INSTANCES_H

// For SharedQueueWrapper class

extern SharedQueueWrapper
    g_tensor_layer_queue; // For weights/tensor layer addresses
extern SharedQueueWrapper g_kv_cache_queue; // For KV cache addresses
extern SharedTimingInfoWrapper g_timing_info_var;
#endif // SHARED_QUEUE_INSTANCES_H
