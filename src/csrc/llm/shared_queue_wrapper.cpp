#include "shared_queue_wrapper.h"
#include <condition_variable>
#include <cstdio> // for printf
#include <cstring>
#include <mutex>
#include "shared_queue.h"

SharedTimingInfoWrapper::SharedTimingInfoWrapper() {
  shared_item.token = 0;
  shared_item.layer = 0;
  shared_item.ts = 0;
}

void SharedTimingInfoWrapper::read(SharedTimingInfo* item) {
  std::unique_lock<std::mutex> lock(timing_mtx);
  *item = shared_item;
  lock.unlock();
}

void SharedTimingInfoWrapper::write(const SharedTimingInfo& item) {
  std::unique_lock<std::mutex> lock(timing_mtx);
  shared_item.token = item.token;
  shared_item.layer = item.layer;
  shared_item.ts = item.ts;
  lock.unlock();
}

SharedQueueWrapper::SharedQueueWrapper() {
  queue_.head = 0;
  queue_.tail = 0;
}

void SharedQueueWrapper::enqueue(const TensorLayerAddressRange& item) {
  std::unique_lock<std::mutex> lock(mtx_);
  size_t next_head = (queue_.head + 1) % SHARED_QUEUE_CAPACITY;
  cv_.wait(
      lock, [&] { return next_head != queue_.tail; }); // wait if queue is full

  queue_.buffer[queue_.head] = item;
  queue_.head = next_head;

  // // Print the address of the SharedQueueWrapper object
  // printf("[SharedQueueWrapper::enqueue] SharedQueueWrapper object address:
  // %p\n", (void*)this);

  // // Print the address of the passed item object
  // printf("[SharedQueueWrapper::enqueue] Item object address passed in: %p\n",
  // (const void*)&item);

  lock.unlock();
  cv_.notify_one();
}

bool SharedQueueWrapper::dequeue(TensorLayerAddressRange* item) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (queue_.head == queue_.tail)
    return false; // empty
  *item = queue_.buffer[queue_.tail];
  queue_.tail = (queue_.tail + 1) % SHARED_QUEUE_CAPACITY;
  lock.unlock();
  cv_.notify_one();
  return true;
}

void SharedQueueWrapper::wait_and_dequeue(TensorLayerAddressRange* item) {
  // Print the address of the SharedQueueWrapper object (the queue instance)
  // printf("[SharedQueueWrapper::wait_and_dequeue] SharedQueueWrapper object
  // address: %p\n", (void*)this);

  std::unique_lock<std::mutex> lock(mtx_);
  cv_.wait(lock, [&] { return queue_.head != queue_.tail; }); // wait if empty
  *item = queue_.buffer[queue_.tail];
  queue_.tail = (queue_.tail + 1) % SHARED_QUEUE_CAPACITY;
  lock.unlock();
  cv_.notify_one();
}
