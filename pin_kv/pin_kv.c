#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <sys/wait.h>

#define MAX_LAYERS  33  // 1-based index, up to 32
#define PAGE_SIZE   4096UL

typedef struct {
    uint64_t k_start, k_end;
    uint64_t v_start, v_end;
    size_t k_size, v_size;
    int valid;
} layer_info_t;

int move_pages_wrapper(pid_t pid, unsigned long count, void **pages, const int *nodes, int *status, int flags) {
    // Use syscall directly because glibc may not expose move_pages for remote PIDs
    return syscall(__NR_move_pages, pid, count, pages, nodes, status, flags);
}

void migrate_range(pid_t pid, uint64_t start, size_t size, int node) {
    size_t npages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
    void **pages = malloc(npages * sizeof(void*));
    int *nodes = malloc(npages * sizeof(int));
    int *status = malloc(npages * sizeof(int));
    if (!pages || !nodes || !status) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    for (size_t i = 0; i < npages; ++i) {
        pages[i] = (void*)(start + i * PAGE_SIZE);
        nodes[i] = node;
    }
    int ret = move_pages_wrapper(pid, npages, pages, nodes, status, MPOL_MF_MOVE);
    if (ret < 0) {
        perror("move_pages");
        fprintf(stderr, "  (address range: 0x%lx - 0x%lx)\n", start, start + size);
    } else {
        printf("Migrated %zu pages (0x%lx - 0x%lx) to node %d for pid %d\n", npages, start, start + size, node, pid);
    }
    free(pages);
    free(nodes);
    free(status);
}

int wait_for_kv_dump(const char* log_file) {
    FILE* f;
    char line[512];
    while (1) {
        f = fopen(log_file, "r");
        if (!f) {
            perror("fopen");
            sleep(1);
            continue;
        }
        int found = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "KV dump over")) {
                found = 1;
                break;
            }
        }
        fclose(f);
        if (found) {
            printf("Found 'KV dump over'. Proceeding to parse and migrate.\n");
            return 0;
        }
        // sleep(1); // Wait before checking again
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <pid> <kv_log_file> <layer_id (0 for all, 1-32 for specific)>\n", argv[0]);
        return 1;
    }

    pid_t pid = (pid_t)atoi(argv[1]);
    const char* log_file = argv[2];
    int target_layer = atoi(argv[3]);

    // Wait for "KV dump over" to appear in the log file
    wait_for_kv_dump(log_file);

    // Now parse the log file
    FILE* f = fopen(log_file, "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    layer_info_t layers[MAX_LAYERS] = {0};
    char line[512];
    int current_layer = -1;

    while (fgets(line, sizeof(line), f)) {
        int lnum;
        if (sscanf(line, "Layer %d:", &lnum) == 1) {
            current_layer = lnum;
            if (current_layer >= 1 && current_layer <= 32) {
                layers[current_layer].valid = 1;
            }
            continue;
        }
        if (current_layer >= 1 && current_layer <= 32) {
            uint64_t start, end;
            size_t size;
            if (sscanf(line, "Key tensor:   start=%lx  end=%lx  size=%zu bytes", &start, &end, &size) == 3) {
                layers[current_layer].k_start = start;
                layers[current_layer].k_end = end;
                layers[current_layer].k_size = size;
            } else if (sscanf(line, "Value tensor: start=%lx  end=%lx  size=%zu bytes", &start, &end, &size) == 3) {
                layers[current_layer].v_start = start;
                layers[current_layer].v_end = end;
                layers[current_layer].v_size = size;
            }
        }
    }
    fclose(f);

    int migrate_all = (target_layer == 0);
    for (int i = 1; i <= 32; ++i) {
        if (!layers[i].valid)
            continue;
        if (migrate_all || i == target_layer) {
            printf("Layer %d:\n", i);
            printf("  K: 0x%lx - 0x%lx (%zu bytes)\n", layers[i].k_start, layers[i].k_end, layers[i].k_size);
            printf("  V: 0x%lx - 0x%lx (%zu bytes)\n", layers[i].v_start, layers[i].v_end, layers[i].v_size);
            migrate_range(pid, layers[i].k_start, layers[i].k_size, 2);
            migrate_range(pid, layers[i].v_start, layers[i].v_size, 2);
        }
    }

    return 0;
}
