/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#include "shm_coll.h"
#include <omp.h>
#include <sys/shm.h>
#include "utils.h"
#include "xsmm_functors.h"

#define BS 512
static const long TPP_SHM_BUF_SIZE =
    env2int("TPP_SHM_BUF_SIZE", 64 * 1024 * 1024);
// Using master port to distinguist multiple distributed instances for setting
// up shared memory
static const long master_port = env2int("MASTER_PORT", 0);
using namespace tpp;
namespace shm_tpp {
template <typename T, int S = BS>
struct TppOps {
  CpyTPP<T> cpy_tpp = CpyTPP<T>(BS);
  ConvertTPP<T, float> ucvt_tpp = ConvertTPP<T, float>(BS);
  ConvertTPP<float, T> dcvt_tpp = ConvertTPP<float, T>(BS);
  AddTPP<float, float, T> add_tpp = AddTPP<float, float, T>(BS);
};

template <typename T>
static TppOps<T> getOps() {}

template <>
TppOps<float> getOps<float>() {
  static TppOps<float> ops_f;
  return ops_f;
}
template <>
TppOps<bfloat16> getOps<bfloat16>() {
  static TppOps<bfloat16> ops_bf;
  return ops_bf;
}
template <>
TppOps<half> getOps<half>() {
  static TppOps<half> ops_hf;
  return ops_hf;
}
} // namespace shm_tpp

class SHMBuffer {
 public:
  static int SHMID;
  static int BARID;
  static const int MAX_RANKS = 64;
  static const int DIRECT_THRESHOLD = 32 * 1024;
  c10::intrusive_ptr<c10d::ProcessGroup> pg;
  int rank;
  int size;
  size_t bufsz;
  int shmid[MAX_RANKS];
  int barid;
  void* shm_data[MAX_RANKS];
  void* scratch_data[MAX_RANKS];
  void* bar_data;
  volatile int* bar1;
  volatile int* bar2;

  SHMBuffer(size_t bufsz_, c10::intrusive_ptr<c10d::ProcessGroup> pg) : pg(pg) {
    bufsz = ((bufsz_ + 4095) / 4096) * 4096 * 2;
    rank = pg->getRank();
    size = pg->getSize();
    /* each process creates its own shared memory */
    // printf("SHM At %d r: %d  s: %d\n", __LINE__, rank, size);
    shmid[rank] = shmget(SHMID + rank, bufsz, IPC_CREAT | 0666);
    TPP_ASSERT(
        shmid[rank] >= 0,
        "shmid cannot create shared memory of size %lu\n",
        bufsz);
    if (rank == 0) {
      barid = shmget(BARID, 4096, IPC_CREAT | 0666);
      TPP_ASSERT(barid >= 0, "barid cannot create shared memory");
    }
    pg->barrier()->wait();
    /* each process attaches itself with other processes */
    for (int i = 0; i < size; i++) {
      if (i != rank)
        shmid[i] = shmget(SHMID + i, bufsz, 0666);
      TPP_ASSERT(shmid[i] >= 0, "shmid cannot get shared memory\n");
    }
    if (rank != 0) {
      barid = shmget(BARID, 4096, IPC_CREAT | 0666);
      TPP_ASSERT(barid >= 0, "barid cannot create shared memory\n");
    }
    for (int i = 0; i < size; i++) {
      shm_data[i] = shmat(shmid[i], NULL, 0);
      TPP_ASSERT(shm_data[i], "shmat failed\n");
      scratch_data[i] = (void*)((char*)shm_data[i] + bufsz / 2);
    }
    bar_data = shmat(barid, NULL, 0);
    TPP_ASSERT(bar_data, "barat failed\n");
    bar1 = (int*)bar_data;
    *bar1 = 0;
    bar2 = bar1 + 128;
    *bar2 = 0;
    pg->barrier()->wait();
    shmctl(shmid[rank], IPC_RMID, NULL);
    shmctl(barid, IPC_RMID, NULL);
    printf("Shm buffer allocated with size %lu\n", bufsz);
  }

  void cleanup_shm() {
    // We can't use pg->barrier here as it may not be available
    for (int i = 0; i < size; i++)
      shmdt(shm_data[i]);
    shmdt(bar_data);
    // printf("SHM buffers deleted\n");
  }

  ~SHMBuffer() {
    cleanup_shm();
  }

  static SHMBuffer* getInst(
      size_t sz,
      c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    static size_t buf_sz = 0;
    static SHMBuffer* inst = nullptr;

    // TODO: check for same pg as well
    if (buf_sz < sz) {
      if (inst != nullptr) {
        delete inst;
        inst = nullptr;
      }
      inst = new SHMBuffer(sz, pg);
      TPP_ASSERT(inst != nullptr, "Unable to create shm buffer\n");
      buf_sz = sz;
    }
    return inst;
  }

  void barrier() {
    static uint32_t count = 0;
    if (count % 2) {
      __sync_fetch_and_add(bar1, 1);
      while ((*bar1 % size) != 0)
        ;
    } else {
      __sync_fetch_and_add(bar2, 1);
      while ((*bar2 % size) != 0)
        ;
    }
    count++;
  }

  at::Tensor getTensor(at::Tensor t) {
    size_t sz = t.numel() * t.element_size();
    TPP_ASSERT(sz <= bufsz, "Requested tensor size too big\n");
    auto ptr = shm_data[rank];
    auto t_new = torch::from_blob(ptr, t.sizes(), t.options());
    return t_new;
  }

  template <typename T>
  void allreduce_impl(at::Tensor t) {
    auto numel = t.numel();
    auto nBytes = numel * t.element_size();
    TPP_ASSERT((size_t)nBytes <= bufsz / 2, "Too large allreduce size");
    long nBlk = (numel + BS - 1) / BS;
    long max_threads = omp_get_max_threads();
    int nThreads = std::min(nBlk, max_threads);
    T* ptr = (T*)t.data_ptr();
    long rem = numel % BS;
    long numel_aligned = numel - rem;
    bool need_copy = ptr != shm_data[rank];
    auto ops = shm_tpp::getOps<T>();
    auto& cpy_tpp = ops.cpy_tpp;
    auto& ucvt_tpp = ops.ucvt_tpp;
    auto& dcvt_tpp = ops.dcvt_tpp;
    auto& add_tpp = ops.add_tpp;

    if (need_copy) {
      auto src = ptr;
      auto dst = (T*)shm_data[rank];
#pragma omp parallel for num_threads(nThreads)
      for (int i = 0; i < numel_aligned; i += BS) {
        cpy_tpp(src + i, dst + i);
      }
      if (rem > 0) {
        for (int i = numel_aligned; i < numel; i++) {
          dst[i] = src[i];
        }
      }
    }

    barrier();

    if (numel <= DIRECT_THRESHOLD) {
      auto dst = (T*)scratch_data[rank];
      auto lsrc = (T*)shm_data[rank];
#pragma omp parallel for num_threads(nThreads)
      for (int i = 0; i < numel; i += BS) {
        float ldst[BS];
        ucvt_tpp(lsrc + i, ldst);
        for (int r = 1; r < size; r++) {
          int r1 = (r + rank) % size;
          auto src = (T*)shm_data[r1];
          add_tpp(ldst, src + i, ldst);
        }
        dcvt_tpp(ldst, dst + i);
      }
      barrier();

      if (true) {
        auto src = (T*)scratch_data[rank];
        auto dst = ptr;
#pragma omp parallel for num_threads(nThreads)
        for (int i = 0; i < numel_aligned; i += BS) {
          cpy_tpp(src + i, dst + i);
        }
        if (rem > 0) {
          for (int i = numel_aligned; i < numel; i++) {
            dst[i] = src[i];
          }
        }
      }
    } else {
      int slice_start = (nBlk * rank / size) * BS;
      int slice_end = (nBlk * (rank + 1) / size) * BS;

      auto dst = (T*)scratch_data[rank];
      auto lsrc = (T*)shm_data[rank];
#pragma omp parallel for num_threads(nThreads)
      for (int i = slice_start; i < slice_end; i += BS) {
        float ldst[BS];
        ucvt_tpp(lsrc + i, ldst);
        for (int r = 1; r < size; r++) {
          int r1 = (r + rank) % size;
          auto src = (T*)shm_data[r1];
          add_tpp(ldst, src + i, ldst);
        }
        dcvt_tpp(ldst, dst + i);
      }
      barrier();
      if (true) {
        for (int r = 0; r < size; r++) {
          int r1 = (r + rank) % size;
          int slice_start = (nBlk * r1 / size) * BS;
          int slice_end = (nBlk * (r1 + 1) / size) * BS;
          bool handle_last_blk = false;
          if (slice_end > numel) {
            slice_end -= BS;
            handle_last_blk = true;
          }

          auto src = (T*)scratch_data[r1];
          auto dst = ptr;
#pragma omp parallel for num_threads(nThreads)
          for (int i = slice_start; i < slice_end; i += BS) {
            cpy_tpp(src + i, dst + i);
          }
          if (handle_last_blk) {
            for (int i = slice_end; i < numel; i++) {
              dst[i] = src[i];
            }
          }
        }
      }
    }
  }

  void allreduce(at::Tensor t) {
    auto dt = t.dtype();
    if (dt == at::kFloat) {
      allreduce_impl<float>(t);
    } else if (dt == at::kBFloat16) {
      allreduce_impl<bfloat16>(t);
    } else if (dt == at::kHalf) {
      allreduce_impl<half>(t);
    } else {
      TPP_ASSERT(0, "Unsupported dtype in allreduce\n");
    }
  }
};

int SHMBuffer::SHMID = 100 + master_port;
int SHMBuffer::BARID = 10000 + master_port;

void shm_allreduce(
    at::Tensor t_in,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group) {
  if (!process_group) {
    printf("Missing process group when using model parallel, use set_pg()\n");
    exit(1);
  }
  auto shm_inst = SHMBuffer::getInst(TPP_SHM_BUF_SIZE, process_group);
  long max_elem = TPP_SHM_BUF_SIZE / t_in.element_size();
  long numel = t_in.numel();
  if (numel <= max_elem) {
    shm_inst->allreduce(t_in);
  } else {
    t_in = t_in.view({-1});
    for (int64_t i = 0; i < numel; i += max_elem) {
      auto start = i;
      auto end = start + max_elem;
      if (end > numel)
        end = numel;
      auto t = t_in.slice(0, start, end, 1);
      shm_inst->allreduce(t);
    }
  }
}

#undef BS
