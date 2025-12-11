import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ======================
# 1. Read data
# ======================

df = pd.read_csv("download_bandwidth.csv")

# Time as x-axis
time = df["Time (EST)"]

# Regions (order determines stacking order)
regions = [
    "Central America",
    "Africa",
    "Middle East",
    "Oceania",
    "Russia",
    "Asia",
    "South America",
    "Europe",
    "North America"
]

data = [df[r] for r in regions]

# ======================
# 2. Steam style color scheme
# ======================

colors = [
    "#e0702f",  # Central America
    "#cfc7a0",  # Africa
    "#9fc3d1",  # Middle East
    "#c49a4a",  # Oceania
    "#4f6fa1",  # Russia
    "#a84e2a",  # Asia
    "#b7db5a",  # South America
    "#9bb6c3",  # Europe
    "#cfd9df"  # North America
]

# ======================
# 3. Plotting
# ======================

fig, ax = plt.subplots(
    figsize=(14, 7),
    dpi=120,
    facecolor="#1b2838"
)

ax.set_facecolor("#1f1f1f")

ax.stackplot(
    time,
    data,
    colors=colors,
    linewidth=0
)

# ======================
# 4. Axes & grid
# ======================

ax.set_ylim(0, 40000)
ax.set_yticks([0, 10000, 20000, 30000, 40000])
ax.set_yticklabels(
    ["0 Gbps", "10,000 Gbps", "20,000 Gbps", "30,000 Gbps", "40,000 Gbps"],
    color="#8f98a0"
)

# Modify x-axis: display every 6 hours
total_points = len(time)
step = max(1, total_points // 8)  # 48 hours / 6 hours = 8 ticks
xtick_indices = list(range(0, total_points, step))
if xtick_indices[-1] != total_points - 1:
    xtick_indices.append(total_points - 1)

ax.set_xticks(xtick_indices)
ax.set_xticklabels([time.iloc[i] for i in xtick_indices], rotation=0, ha='center')

ax.tick_params(axis="x", colors="#8f98a0")
ax.tick_params(axis="y", colors="#8f98a0")

ax.grid(axis="y", color="#2a2a2a", linewidth=1)

for spine in ax.spines.values():
    spine.set_visible(False)

# ======================
# 5. Title
# ======================

ax.set_title(
    "Steam Download Bandwidth Usage (Last 48 Hours)",
    fontsize=18,
    color="#8dc3ff",
    pad=20,
    loc="center"
)

# ======================
# 6. Annotate peak
# ======================

# Calculate total bandwidth
total_bandwidth = sum(data)
peak_idx = np.argmax(total_bandwidth)
peak_value = total_bandwidth[peak_idx]
peak_time = time.iloc[peak_idx]

# Add marker at peak position
ax.plot(peak_idx, peak_value, 'o', color='#ff4444', markersize=8, zorder=5)

# Intelligently adjust annotation position to avoid overlapping with title
if peak_idx < total_points * 0.3:
    # Peak on left, annotation on right
    xytext_offset = (20, -30)
elif peak_idx > total_points * 0.7:
    # Peak on right, annotation on left
    xytext_offset = (-20, -30)
else:
    # Peak in middle, annotation below
    xytext_offset = (0, -50)

ax.annotate(
    f'Peak: {peak_value:,.0f} Gbps\n{peak_time}',
    xy=(peak_idx, peak_value),
    xytext=xytext_offset,
    textcoords='offset points',
    color='#ff6666',
    fontsize=10,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#1b2838', edgecolor='#ff4444', linewidth=1.5),
    arrowprops=dict(arrowstyle='->', color='#ff4444', lw=1.5)
)

# ======================
# 7. Add legend (top right corner)
# ======================

legend_x = 0.98
legend_y = 0.98
legend_spacing = 0.04

for i, (region, color) in enumerate(zip(regions, colors)):
    y_pos = legend_y - i * legend_spacing

    # Color square
    rect = Rectangle(
        (legend_x - 0.03, y_pos - 0.012),
        0.02, 0.024,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor='none'
    )
    ax.add_patch(rect)

    # Text label
    ax.text(
        legend_x - 0.035,
        y_pos,
        region,
        transform=ax.transAxes,
        fontsize=9,
        color='#8f98a0',
        ha='right',
        va='center'
    )

# ======================
# 8. Save
# ======================

plt.tight_layout()
plt.savefig(
    "steam_download_bandwidth.png",
    dpi=300,
    facecolor=fig.get_facecolor(),
    bbox_inches="tight"
)

plt.show()
