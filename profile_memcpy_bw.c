#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numa.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>

#define PAGE_SIZE 4096

typedef struct {
    void *src;
    void *dst;
    size_t size;
} thread_arg_t;

void *thread_memcpy(void *arg) {
    thread_arg_t *targ = (thread_arg_t *)arg;
    memcpy(targ->dst, targ->src, targ->size);
    return NULL;
}

static double get_elapsed_time(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

void format_size(size_t bytes, double *value, const char **unit) {
    if (bytes >= (1UL << 30)) {
        *value = bytes / (1024.0 * 1024.0 * 1024.0);
        *unit = "GB";
    } else if (bytes >= (1UL << 20)) {
        *value = bytes / (1024.0 * 1024.0);
        *unit = "MB";
    } else if (bytes >= (1UL << 10)) {
        *value = bytes / 1024.0;
        *unit = "KB";
    } else {
        *value = bytes;
        *unit = "bytes";
    }
}

void format_bandwidth(double bytes_per_sec, double *value, const char **unit) {
    if (bytes_per_sec >= 1e9) {
        *value = bytes_per_sec / 1e9;
        *unit = "GB/s";
    } else if (bytes_per_sec >= 1e6) {
        *value = bytes_per_sec / 1e6;
        *unit = "MB/s";
    } else if (bytes_per_sec >= 1e3) {
        *value = bytes_per_sec / 1e3;
        *unit = "KB/s";
    } else {
        *value = bytes_per_sec;
        *unit = "B/s";
    }
}

int main() {
    if (numa_available() < 0) {
        fprintf(stderr, "NUMA is not available on this system.\n");
        return 1;
    }

    FILE *log_file = fopen("memcpy_migration_bandwidth_profile.log", "w");
    if (!log_file) {
        perror("Failed to open log file");
        return 1;
    }

    int thread_counts[] = {1, 2, 4, 8};
    size_t max_size = 1UL << 30; // 1 GB

    for (size_t t = 0; t < sizeof(thread_counts)/sizeof(thread_counts[0]); t++) {
        int num_threads = thread_counts[t];
        fprintf(log_file, "\nThread count: %d\n", num_threads);
        fprintf(stdout, "\nThread count: %d\n", num_threads);

        for (size_t pages = 1; pages <= max_size / PAGE_SIZE; pages *= 4) {
            size_t size = pages * PAGE_SIZE;
            void *buf_node0 = numa_alloc_onnode(size, 0);
            void *buf_node2 = numa_alloc_onnode(size, 2);

            if (!buf_node0 || !buf_node2) {
                fprintf(log_file, "  Allocation failed for %zu bytes\n", size);
                fprintf(stderr, "  Allocation failed for %zu bytes\n", size);
                continue;
            }

            // Fill source buffer
            memset(buf_node0, 0xAA, size);

            pthread_t threads[num_threads];
            thread_arg_t args[num_threads];
            size_t chunk_size = size / num_threads;

            struct timeval start, end;
            gettimeofday(&start, NULL);

            // Create threads for parallel memcpy
            for (int i = 0; i < num_threads; i++) {
                args[i].src = (char *)buf_node0 + i * chunk_size;
                args[i].dst = (char *)buf_node2 + i * chunk_size;
                args[i].size = (i == num_threads - 1) ? (size - i * chunk_size) : chunk_size;
                pthread_create(&threads[i], NULL, thread_memcpy, &args[i]);
            }

            // Wait for all threads
            for (int i = 0; i < num_threads; i++) {
                pthread_join(threads[i], NULL);
            }

            gettimeofday(&end, NULL);

            double elapsed = get_elapsed_time(start, end);
            double bandwidth_bytes_per_sec = size / elapsed;

            // Format size
            double formatted_size;
            const char *size_unit;
            format_size(size, &formatted_size, &size_unit);

            // Format bandwidth
            double formatted_bw;
            const char *bw_unit;
            format_bandwidth(bandwidth_bytes_per_sec, &formatted_bw, &bw_unit);

            fprintf(log_file, "  Size: %7.2f %-5s | Bandwidth: %8.2f %s\n",
                    formatted_size, size_unit, formatted_bw, bw_unit);
            printf("  Size: %7.2f %-5s | Bandwidth: %8.2f %s\n",
                   formatted_size, size_unit, formatted_bw, bw_unit);

            numa_free(buf_node0, size);
            numa_free(buf_node2, size);
        }
    }

    fclose(log_file);
    return 0;
}
