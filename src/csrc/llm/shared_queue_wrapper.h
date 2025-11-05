#ifndef SHARED_QUEUE_WRAPPER_H
#define SHARED_QUEUE_WRAPPER_H

#include <condition_variable>
#include <mutex>
#include "shared_queue.h" // For TensorLayerAddressRange, MAX constants

class SharedQueueWrapper {
 public:
  SharedQueueWrapper();

  void enqueue(const TensorLayerAddressRange& item);
  bool dequeue(TensorLayerAddressRange* item);
  void wait_and_dequeue(TensorLayerAddressRange* item);

 private:
  SharedQueue queue_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

class SharedTimingInfoWrapper {
 public:
  SharedTimingInfoWrapper(); // Constructor name fixed to class name

  void write(const SharedTimingInfo& item);
  void read(
      SharedTimingInfo* item); // Changed pointer to reference for safer copying

 private:
  SharedTimingInfo shared_item;
  std::mutex timing_mtx;
};

#endif // SHARED_QUEUE_WRAPPER_H
