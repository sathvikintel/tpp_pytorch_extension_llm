// shared_queue_c_wrapper.h

#ifndef SHARED_QUEUE_C_WRAPPER_H
#define SHARED_QUEUE_C_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define MAX_NAME_LEN 64

typedef struct {
    char name[MAX_NAME_LEN];
    size_t layer_num;
    uintptr_t start_addr;
    uintptr_t end_addr;
} TensorLayerAddressRange;

/**
 * Blocks until an item from the shared C++ queue is available.
 * Copies the item into `output`.
 * Returns 1 on success, 0 on failure (e.g., null pointer).get
 */
int blocking_dequeue(TensorLayerAddressRange* output);

typedef struct {
    size_t token;
    size_t layer;
    long long ts;
} SharedTimingInfo;

int get_timing_info(SharedTimingInfo* output);

#ifdef __cplusplus
}
#endif

#endif // SHARED_QUEUE_C_WRAPPER_H

