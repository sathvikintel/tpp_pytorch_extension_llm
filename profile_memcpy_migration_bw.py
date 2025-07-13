import sys
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(log_file):
    """Parse the log file and extract bandwidth data."""
    with open(log_file, 'r') as f:
        log_data = f.read()
    
    thread_pattern = re.compile(r'Thread count: (\d+)')
    data_pattern = re.compile(
        r'Size:\s+([\d.]+)\s+(\w+).*?Bandwidth:\s+([\d.]+)\s+(\w+)/s'
    )
    
    results = {}
    current_thread = None
    
    for line in log_data.splitlines():
        thread_match = thread_pattern.match(line)
        if thread_match:
            current_thread = int(thread_match.group(1))
            results[current_thread] = {'sizes': [], 'bandwidths': []}
            continue
            
        data_match = data_pattern.search(line)
        if data_match and current_thread:
            size_val = float(data_match.group(1))
            size_unit = data_match.group(2)
            bw_val = float(data_match.group(3))
            bw_unit = data_match.group(4)
            
            # Convert size to bytes
            size_multiplier = {
                'KB': 1024,
                'MB': 1024**2,
                'GB': 1024**3
            }.get(size_unit, 1)
            size_bytes = size_val * size_multiplier
            
            # Convert bandwidth to GB/s
            bw_multiplier = {
                'KB': 1/(1024**2),
                'MB': 1/1024,
                'GB': 1
            }.get(bw_unit, 0)
            bw_gb = bw_val * bw_multiplier
            
            results[current_thread]['sizes'].append(size_bytes)
            results[current_thread]['bandwidths'].append(bw_gb)
            
    return results

def create_aesthetic_plot(data, output_file):
    """Create an aesthetic line plot of the bandwidth data without any grid lines."""
    sns.set_theme(style="white", context="notebook", font_scale=1.2)
    plt.figure(figsize=(12, 7))
    
    palette = sns.color_palette("husl", len(data))
    
    for i, (thread_count, values) in enumerate(data.items()):
        sizes = values['sizes']
        bandwidths = values['bandwidths']
        sizes_mb = [s / (1024**2) for s in sizes]
        
        plt.plot(
            sizes_mb, 
            bandwidths, 
            marker='o', 
            markersize=8,
            linewidth=2.5,
            label=f'{thread_count} Threads',
            color=palette[i]
        )
    
    plt.xscale('log')
    x_ticks = [4/1024, 16/1024, 64/1024, 256/1024, 1, 4, 16, 64, 256, 1024]
    x_labels = [
        '4KB', '16KB', '64KB', '256KB', '1MB', '4MB', '16MB', '64MB', '256MB', '1GB'
    ]
    plt.xticks(x_ticks, x_labels, rotation=45)
    
    plt.xlabel('Transfer Size', fontweight='bold', fontsize=14)
    plt.ylabel('Bandwidth (GB/s)', fontweight='bold', fontsize=14)
    plt.title('Memcpy Migration Bandwidth Analysis', fontweight='bold', fontsize=16)
    
    # No grid lines at all
    plt.grid(False)
    
    plt.legend(title='Thread Count', title_fontsize='13', fontsize='12')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return os.path.abspath(output_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_bandwidth.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = "migration_bandwidth_plot.png"
    
    bandwidth_data = parse_log_file(log_file)
    abs_path = create_aesthetic_plot(bandwidth_data, output_file)
    print(abs_path)
