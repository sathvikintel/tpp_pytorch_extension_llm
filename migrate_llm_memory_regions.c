#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <numaif.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>

#define HBM_NODE 2
#define LINE_MAX 256
#define PID_CHECK_INTERVAL 10  // Check every 10 iterations (1 second)

typedef struct {
    pid_t pid;
    void* start;
    void* end;
    int node;
    FILE* logf;
    size_t thread_idx;
    size_t nthreads;
} migrate_thread_arg_t;

static int process_exists(pid_t pid) {
    if (kill(pid, 0) == 0) return 1;
    return (errno != ESRCH);
}

// Thread function to migrate a subrange
void* migrate_range_thread(void* arg_) {
    migrate_thread_arg_t* arg = (migrate_thread_arg_t*)arg_;
    long page_size = sysconf(_SC_PAGESIZE);
    size_t total_length = (char*)arg->end - (char*)arg->start;
    size_t total_pages = (total_length + page_size - 1) / page_size;

    // Compute subrange for this thread
    size_t pages_per_thread = total_pages / arg->nthreads;
    size_t extra = total_pages % arg->nthreads;
    size_t my_pages = pages_per_thread + (arg->thread_idx < extra ? 1 : 0);
    size_t my_start_idx = arg->thread_idx * pages_per_thread + (arg->thread_idx < extra ? arg->thread_idx : extra);
    void* my_start = (char*)arg->start + my_start_idx * page_size;
    void* my_end = (char*)my_start + my_pages * page_size;
    if (my_end > arg->end) my_end = arg->end;
    size_t my_length = (char*)my_end - (char*)my_start;

    if (my_length == 0) return NULL;

    void **pages = malloc(my_pages * sizeof(void*));
    int *nodes = malloc(my_pages * sizeof(int));
    int *status = malloc(my_pages * sizeof(int));
    if (!pages || !nodes || !status) {
        fprintf(stderr, "malloc failed in thread %zu\n", arg->thread_idx);
        return NULL;
    }

    for (size_t i = 0; i < my_pages; ++i) {
        pages[i] = (char*)my_start + i * page_size;
        nodes[i] = arg->node;
    }

    int ret = move_pages(arg->pid, my_pages, pages, nodes, status, MPOL_MF_MOVE);

    time_t now = time(NULL);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    if (ret < 0) {
        fprintf(stderr, "move_pages failed for [%p-%p] in pid %d: %s\n", my_start, my_end, arg->pid, strerror(errno));
        if (arg->logf) {
            fprintf(arg->logf, "[%s] FAIL: [Thread %zu] [%p-%p] in pid %d: %s | size: %.3f MB\n",
                    tbuf, arg->thread_idx, my_start, my_end, arg->pid, strerror(errno),
                    my_length / (1024.0 * 1024.0));
            fflush(arg->logf);
        }
    } else {
        printf("Thread %zu: Migrated %zu pages [%p-%p] in pid %d to node %d\n", arg->thread_idx, my_pages, my_start, my_end, arg->pid, arg->node);
        if (arg->logf) {
            fprintf(arg->logf, "[%s] SUCCESS: [Thread %zu] [%p-%p] in pid %d to node %d | size: %.3f MB\n",
                    tbuf, arg->thread_idx, my_start, my_end, arg->pid, arg->node,
                    my_length / (1024.0 * 1024.0));
            fflush(arg->logf);
        }
    }

    free(pages);
    free(nodes);
    free(status);
    return NULL;
}

// Helper: migrate a memory region in another process to NUMA node 2 using multiple threads
int migrate_range_multithreaded(pid_t pid, void* start, void* end, int node, FILE *logf, size_t nthreads) {
    if (nthreads < 1) nthreads = 1;
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    migrate_thread_arg_t *args = malloc(nthreads * sizeof(migrate_thread_arg_t));
    if (!threads || !args) {
        fprintf(stderr, "malloc failed for threads\n");
        return -1;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (size_t i = 0; i < nthreads; ++i) {
        args[i].pid = pid;
        args[i].start = start;
        args[i].end = end;
        args[i].node = node;
        args[i].logf = logf;
        args[i].thread_idx = i;
        args[i].nthreads = nthreads;
        pthread_create(&threads[i], NULL, migrate_range_thread, &args[i]);
    }
    for (size_t i = 0; i < nthreads; ++i) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    size_t total_length = (char*)end - (char*)start;
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double bandwidth = ((double)total_length / (1024.0 * 1024.0 * 1024.0)) / dt; // GB/s

    time_t now = time(NULL);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    if (logf) {
        fprintf(logf, "[%s] AGGREGATE: Migrated %zu bytes [%p-%p] in pid %d to node %d | time: %.6f s | bandwidth: %.3f GB/s | threads: %zu\n",
                tbuf, total_length, start, end, pid, node, dt, bandwidth, nthreads);
        fflush(logf);
    }

    free(threads);
    free(args);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <llm_pid> <log_file> <migration_log_file> <num_threads>\n", argv[0]);
        return 1;
    }

    pid_t llm_pid = (pid_t)atoi(argv[1]);
    const char *logfile = argv[2];
    const char *migration_logfile = argv[3];
    size_t nthreads = (size_t)atoi(argv[4]);
    if (nthreads < 1) nthreads = 1;

    FILE *fp = fopen(logfile, "r");
    if (!fp) {
        perror("fopen log file");
        return 1;
    }
    FILE *logf = fopen(migration_logfile, "a");
    if (!logf) {
        perror("fopen migration log file");
        fclose(fp);
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    char line[LINE_MAX];
    int check_counter = 0;
    
    while (1) {
        if (fgets(line, sizeof(line), fp)) {
            // Process line
            unsigned long start, end;
            if (sscanf(line, "%lx-%lx", &start, &end) == 2) {
                migrate_range_multithreaded(llm_pid, (void*)start, (void*)end, HBM_NODE, logf, nthreads);
            }
        } else {
            // Check PID existence every PID_CHECK_INTERVAL cycles
            if (++check_counter >= PID_CHECK_INTERVAL) {
                check_counter = 0;
                if (!process_exists(llm_pid)) {
                    time_t now = time(NULL);
                    char tbuf[64];
                    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
                    fprintf(logf, "[%s] INFO: Workload PID %d exited. Daemon terminating.\n", tbuf, llm_pid);
                    fflush(logf);
                    break;  // Exit loop
                }
            }
            usleep(100000);  // 100 ms
            clearerr(fp);
        }
    }

    // Cleanup
    fclose(fp);
    fclose(logf);
    printf("Daemon exiting. Workload PID %d no longer exists.\n", llm_pid);
    return 0;
}
