import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# ======================
# Steam style configuration (close to official website)
# ======================

class SpeedMapConfig:
    # Ocean / overall background: dark gray-blue
    OCEAN_COLOR = "#303238"     # Dark gray-blue, close to Steam map background

    # Land base: almost pure black
    LAND_BASE = "#050505"       # Continent base color (black)
    # No data area: slightly brighter than land
    NO_DATA = "#151515"

    # Speed gradient colors (green-yellow system - faster is brighter)
    SPEED_COLORS = [
        "#1a2818",  # Very slow - deep dark green (almost black)
        "#2d4028",  # Slow - dark green
        "#405a30",  # Medium slow - ink green
        "#5a7a40",  # Medium - grass green
        "#7aa050",  # Fast - bright green
        "#a8c96e",  # Very fast - light green
        "#d4f57d",  # Extremely fast - fluorescent yellow-green
        "#f0ff90"   # Super fast - extremely bright yellow-green
    ]

    # Border style
    EDGE_SLOW = "#20252c"       # Slightly bright dark gray border
    EDGE_FAST = "#c7d934"       # High-speed country border (bright yellow-green glow)
    EDGE_WIDTH_BASE = 0.3
    EDGE_WIDTH_FAST = 1.2       # High-speed country border thicker and brighter


config = SpeedMapConfig()


# ======================
# 1. Read speed data
# ======================
print("Loading speed data...")

speed = pd.read_csv("speed_by_country_iso3.csv")
speed = speed[speed["iso3"].notna()].copy()
print(f"Speed data: {len(speed)} entities (including countries/territories)")


# ======================
# 2. Load world map (Natural Earth)
# ======================

print("Loading world vector boundaries...")

world = None
try:
    # 50m precision
    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
    )
    print("Using 1:50m high-precision country boundaries")
except Exception:
    try:
        # 110m precision
        world = gpd.read_file(
            "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        )
        print("Using 1:110m medium-precision country boundaries")
    except Exception:
        # Fallback
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        print("Warning: Using built-in lowres world map")

# Unify ISO3 column name
if 'ISO_A3' in world.columns:
    world['iso_a3'] = world['ISO_A3']
elif 'iso_a3' not in world.columns and 'ADM0_A3' in world.columns:
    world['iso_a3'] = world['ADM0_A3']

# Unify country name column (for printing)
if 'ADMIN' in world.columns:
    world_name_col = 'ADMIN'
elif 'name' in world.columns:
    world_name_col = 'name'
else:
    world_name_col = world.columns[0]

print(f"Map data: {len(world)} countries/regions")


# ======================
# 3. Merge speed data to map
# ======================

merged = world.merge(speed, left_on="iso_a3", right_on="iso3", how="left")
has_data = merged["speed_mbps"].notna().sum()
print(f"Countries/regions with speed data: {has_data}")

valid_speeds = merged["speed_mbps"].dropna()
if len(valid_speeds) > 0:
    vmin = np.percentile(valid_speeds, 2)
    vmax = np.percentile(valid_speeds, 98)
    print(f"Speed color scale range (2%-98%): {vmin:.1f} - {vmax:.1f} Mbps")
else:
    vmin, vmax = 0, 100

# Mark high-speed countries (top 25%)
if len(valid_speeds) > 0:
    speed_threshold = np.percentile(valid_speeds, 75)
    merged["is_fast"] = merged["speed_mbps"] >= speed_threshold
else:
    merged["is_fast"] = False


# ======================
# 4. Steam style color scale
# ======================

cmap = LinearSegmentedColormap.from_list(
    "steam_speed",
    config.SPEED_COLORS,
    N=256
)


# ======================
# 5. Annotation function: draw line from country point to text box at specified coordinates
# ======================

def annotate_country_line_data(ax, row, text, text_lon, text_lat):
    """
    text_lon, text_lat: Text box placement position (longitude, latitude).
    Draw a line from country representative point -> text box, and draw highlight box around text box.
    """
    geom = row.geometry
    rep_pt = geom.representative_point()
    x, y = rep_pt.x, rep_pt.y

    ax.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=(text_lon, text_lat),
        textcoords="data",
        fontsize=11,
        color="#f0ff90",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="-",
            color="#f0ff90",
            linewidth=1.5,
            alpha=0.9,
            connectionstyle="arc3,rad=0.15",
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#1a1f26",
            edgecolor="#f0ff90",
            linewidth=1.0,
            alpha=0.9
        )
    )


# ======================
# 6. Draw map
# ======================

print("Drawing Steam style map...")

fig, ax = plt.subplots(
    figsize=(26, 13),
    facecolor=config.OCEAN_COLOR,
    dpi=100
)
ax.set_facecolor(config.OCEAN_COLOR)

# Base black land
world.plot(
    ax=ax,
    color=config.LAND_BASE,
    edgecolor=config.EDGE_SLOW,
    linewidth=0.15,
    alpha=1.0
)

# Countries with data (color layer)
merged_with_data = merged[merged["speed_mbps"].notna()]
merged_with_data.plot(
    ax=ax,
    column="speed_mbps",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    edgecolor=config.EDGE_SLOW,
    linewidth=0.2,
    alpha=0.9
)

# High-speed country border highlight
fast_countries = merged[merged["is_fast"] == True]
if len(fast_countries) > 0:
    fast_countries.plot(
        ax=ax,
        facecolor="none",
        edgecolor=config.EDGE_FAST,
        linewidth=config.EDGE_WIDTH_FAST,
        alpha=0.6
    )
    print(f"Highlighted {len(fast_countries)} high-speed countries")

# Countries with no data
merged_no_data = merged[merged["speed_mbps"].isna()]
merged_no_data.plot(
    ax=ax,
    color=config.NO_DATA,
    edgecolor=config.EDGE_SLOW,
    linewidth=0.15,
    alpha=0.8
)


# ======================
# 7. Annotations: Myanmar (slowest) + South Korea (fastest) + United States (value only)
# ======================

# --- Myanmar (Min) ---
mm = merged[(merged["iso3"] == "MMR") & merged["speed_mbps"].notna()]
if not mm.empty:
    r_mm = mm.iloc[0]
    text_min = f"Min {r_mm['speed_mbps']:.1f} Mbps\nMyanmar"
    annotate_country_line_data(
        ax,
        r_mm,
        text_min,
        text_lon=85,    # Indian Ocean
        text_lat=-20
    )
    print(f"Annotated slowest country: Myanmar ({r_mm['speed_mbps']:.1f} Mbps)")
else:
    print("Warning: Myanmar (iso3 = MMR) not found in merged, skipping minimum annotation")

# --- South Korea (Max) ---
kr = merged[(merged["iso3"] == "KOR") & merged["speed_mbps"].notna()]
if not kr.empty:
    r_kr = kr.iloc[0]
    text_max = f"Max {r_kr['speed_mbps']:.1f} Mbps\nSouth Korea"
    annotate_country_line_data(
        ax,
        r_kr,
        text_max,
        text_lon=160,   # Slightly to the left
        text_lat=20
    )
    print(f"Annotated fastest country: South Korea ({r_kr['speed_mbps']:.1f} Mbps)")
else:
    if len(valid_speeds) > 0:
        max_idx = merged["speed_mbps"].idxmax()
        r_max = merged.loc[max_idx]
        if "country" in r_max.index and pd.notna(r_max["country"]):
            c_name_max = r_max["country"]
        else:
            c_name_max = r_max.get(world_name_col, "Top country")
        text_max = f"Max {r_max['speed_mbps']:.1f} Mbps\n{c_name_max}"
        annotate_country_line_data(
            ax,
            r_max,
            text_max,
            text_lon=115,
            text_lat=45
        )
        print(f"Annotated fastest country: {c_name_max} ({r_max['speed_mbps']:.1f} Mbps)")

# --- United States (only annotate value, no Max) ---
us = merged[(merged["iso3"] == "USA") & merged["speed_mbps"].notna()]
if not us.empty:
    r_us = us.iloc[0]
    text_us = f"United States\n{r_us['speed_mbps']:.1f} Mbps"
    # Place text slightly above US in the northwest Pacific
    annotate_country_line_data(
        ax,
        r_us,
        text_us,
        text_lon=-150,   # North Pacific
        text_lat=0
    )
    print(f"Annotated United States: {r_us['speed_mbps']:.1f} Mbps")
else:
    print("Warning: United States (iso3 = USA) not found in merged, skipping US annotation")


# ======================
# 8. Legend
# ======================

legend_elements = [
    mpatches.Patch(color=config.SPEED_COLORS[0], label='< 10 Mbps'),
    mpatches.Patch(color=config.SPEED_COLORS[2], label='10–30 Mbps'),
    mpatches.Patch(color=config.SPEED_COLORS[4], label='30–60 Mbps'),
    mpatches.Patch(color=config.SPEED_COLORS[6], label='60–100 Mbps'),
    mpatches.Patch(color=config.SPEED_COLORS[7], label='> 100 Mbps'),
    mpatches.Patch(color=config.NO_DATA,          label='No data', alpha=0.8)
]

legend = ax.legend(
    handles=legend_elements,
    loc='lower left',
    frameon=True,
    fancybox=False,
    fontsize=11,
    framealpha=0.85
)
legend.get_frame().set_facecolor('#1a1f26')
legend.get_frame().set_edgecolor('#3a4a3a')
legend.get_frame().set_linewidth(1.5)
for text in legend.get_texts():
    text.set_color('#e0e0e0')


# ======================
# 9. Viewport and style
# ======================

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 85)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.title(
    "Global Average Download Speed by Country — Last 7 Days",
    fontsize=24,
    color="#ffffff",
    pad=30,
    fontweight="bold",
    fontfamily="sans-serif"
)

plt.tight_layout()


# ======================
# 10. Save & console output
# ======================

output_file = "steam_speed_map_annotated.png"
plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
    edgecolor="none",
    pad_inches=0.1
)

print(f"\nSaved successfully: {output_file}")
plt.show()

print("\n" + "=" * 60)
print("Speed Analysis")
print("=" * 60)
if len(valid_speeds) > 0:
    print(f"Countries covered: {has_data}")
    print(f"Average speed: {valid_speeds.mean():.2f} Mbps")
    print(f"Median speed: {valid_speeds.median():.2f} Mbps")
    print(f"Speed range: {valid_speeds.min():.1f} - {valid_speeds.max():.1f} Mbps")
    print(f"High-speed countries (top 25%): {(merged['is_fast'] == True).sum()}")

    top5 = merged[merged["speed_mbps"].notna()].nlargest(5, 'speed_mbps')
    print("\nTop 5 fastest countries:")
    for _, row in top5.iterrows():
        if "country" in row.index and pd.notna(row["country"]):
            cname = row["country"]
        else:
            cname = row.get(world_name_col, "Unknown")
            print(f"   {cname}: {row['speed_mbps']:.1f} Mbps")
print("=" * 60)
