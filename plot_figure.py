import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

FAST_MEMORY_SIZE = 1000

# Create a smooth transition using a sigmoid curve
time = np.linspace(0, 1, 100)
x = FAST_MEMORY_SIZE / (1 + np.exp(-12 * (time - 0.5)))  # KV Cache (curve)
y = FAST_MEMORY_SIZE - x                                 # Weights (complementary curve)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.4)
plt.figure(figsize=(12, 6))

plt.plot(time, x, color="#007ACC", linewidth=3, label="KV Cache (x)")
plt.plot(time, y, color="#FF6600", linewidth=3, label="Weights (y)")

plt.title("Fast Memory Consumption of KV Cache and Weights", fontsize=20, weight='bold', pad=15)
plt.xlabel("Time", fontsize=16)
plt.ylabel("Size", fontsize=16)

plt.xticks([])  # Remove x-axis values
plt.yticks([0, FAST_MEMORY_SIZE], ["0", "FAST_MEMORY_SIZE"])  # Only show 0 and FAST_MEMORY_SIZE

plt.legend(frameon=True, fontsize=14, loc='center right')
plt.tight_layout(pad=2.0)

output_file = "kv_cache_weights_curved.png"
plt.savefig(output_file, dpi=220, bbox_inches='tight', transparent=False)
plt.close()

abs_path = os.path.abspath(output_file)
print(f"Plot saved to: {abs_path}")

