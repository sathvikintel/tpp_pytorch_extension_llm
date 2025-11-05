#ifndef SHARED_QUEUE_H
#define SHARED_QUEUE_H

#include <stdint.h>
#include <cstddef>

#define MAX_NAME_LEN 64
#define SHARED_QUEUE_CAPACITY 1024

typedef struct {
  char name[MAX_NAME_LEN];
  size_t layer_num;
  uintptr_t start_addr;
  uintptr_t end_addr;
} TensorLayerAddressRange;

typedef struct {
  TensorLayerAddressRange buffer[SHARED_QUEUE_CAPACITY];
  size_t head;
  size_t tail;
  // For synchronization, use mutex and condvar (defined in C/C++)
} SharedQueue;

typedef struct {
  size_t token;
  size_t layer;
  long long ts;
} SharedTimingInfo;

#endif // SHARED_QUEUE_H
