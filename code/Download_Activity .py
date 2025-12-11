import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import geodatasets


# ======================
# Steam Official Style Configuration
# ======================
class SteamMapConfig:
    # Point density control
    POINT_SCALE = 200.0  # Increased to 200, more dense
    MAX_POINTS_PER_CITY = 1500  # Single city upper limit
    MAX_TOTAL_POINTS = 4_000_000  # Total points upper limit

    # Three-layer point cloud structure (key!)
    CORE_RATIO = 0.40  # Core layer 40%
    MID_RATIO = 0.35  # Middle layer 35%
    OUTER_RATIO = 0.25  # Outer layer 25%

    # Jitter radius (forms natural gradient)
    CORE_JITTER = 0.06  # Very small jitter, super bright core
    MID_JITTER = 0.20  # Medium jitter, bright area
    OUTER_JITTER = 0.50  # Large jitter, sparse diffusion

    # Steam official color scheme
    BG_COLOR = "#1e2d3d"  # Moderate dark blue-gray ocean
    LAND_COLOR = "#0a0d12"  # Dark gray land (almost black)
    EDGE_COLOR = "#0a0d12"  # Almost invisible border
    DOT_COLOR = "#c7d934"  # Steam yellow-green

    # Rendering parameters
    POINT_SIZE = 0.22  # Very small points
    POINT_ALPHA = 0.20  # Very transparent, brightness from overlap

    # Glow enhancement layer
    ENABLE_GLOW_LAYER = True  # Whether to add glow layer
    GLOW_SAMPLE_RATIO = 0.25  # Glow layer samples 25% of points
    GLOW_SIZE = 0.8  # Glow points slightly larger
    GLOW_ALPHA = 0.06  # Glow very transparent


config = SteamMapConfig()

# ======================
# 1. Read standardized data
# ======================
print(" Loading data...")

cities = pd.read_csv("cities_iso3.csv")
traffic = pd.read_csv("traffic_iso3.csv")

# Data cleaning
traffic = traffic[(traffic["iso3"].notna()) & (traffic["country"] != "Unknown")]
cities = cities[cities["iso3"].notna()]

print(f"✓ City data: {len(cities)} rows")
print(f"✓ Traffic data: {len(traffic)} rows")

# Ensure key fields are numeric
traffic["value"] = pd.to_numeric(traffic["value"], errors="coerce")
cities["population"] = pd.to_numeric(cities["population"], errors="coerce")

# ======================
# 2. Unify units to TB
# ======================
print(" Unifying traffic units...")

unit_to_tb = {
    "TB": 1,
    "PB": 1024,
    "GB": 1 / 1024,
    "MB": 1 / 1024 ** 2,
    "KB": 1 / 1024 ** 3,
    "B": 1 / 1024 ** 4,
}

traffic["unit"] = traffic["unit"].astype(str).str.strip()
traffic["traffic_tb"] = traffic["value"] * traffic["unit"].map(unit_to_tb)
traffic = traffic[traffic["traffic_tb"].notna()]

# Aggregate by country
traffic_iso = traffic.groupby("iso3", as_index=False)["traffic_tb"].sum()

print(f"✓ Number of countries after aggregation: {len(traffic_iso)}")
print(f"✓ Total traffic: {traffic_iso['traffic_tb'].sum():.2f} TB")

# ======================
# 3. Distribute to cities by population
# ======================
print("Distributing traffic to cities...")

cities = cities[cities["population"] > 0]

# Geographic coordinate validity check
cities = cities[
    (cities["lat"].between(-90, 90)) &
    (cities["lng"].between(-180, 180))
    ]

merged = cities.merge(traffic_iso, on="iso3", how="inner")

print(f"✓ Number of cities after merge: {len(merged)}")

# Calculate city weights
merged["country_pop"] = merged.groupby("iso3")["population"].transform("sum")
merged = merged[merged["country_pop"] > 0]
merged["pop_weight"] = merged["population"] / merged["country_pop"]
merged["city_traffic_tb"] = merged["traffic_tb"] * merged["pop_weight"]

# Handle outliers
merged["city_traffic_tb"] = merged["city_traffic_tb"].replace([np.inf, -np.inf], np.nan)
merged["city_traffic_tb"] = merged["city_traffic_tb"].fillna(0)

# ======================
# 4. Log compression + point count calculation
# ======================
print("Calculating point cloud density...")

# Log transform to smooth large values
merged["city_traffic_log"] = np.log1p(merged["city_traffic_tb"])
max_log = merged["city_traffic_log"].max()
if max_log > 0:
    merged["city_traffic_log"] = merged["city_traffic_log"] / max_log

# Calculate number of points
num_points_float = merged["city_traffic_log"] * config.POINT_SCALE
num_points_float = num_points_float.clip(lower=0)
merged["num_points"] = np.floor(num_points_float).astype(int)

# Ensure cities with traffic have at least one point
merged.loc[(merged["num_points"] == 0) & (merged["city_traffic_log"] > 0), "num_points"] = 1

# Limit single city upper bound
merged["num_points"] = merged["num_points"].clip(upper=config.MAX_POINTS_PER_CITY)

total_points_est = int(merged["num_points"].sum())
print(f"✓ Estimated total points: {total_points_est:,}")

# ======================
# 5. Generate three-layer point cloud (core algorithm)
# ======================
print("Generating three-layer point cloud structure...")

np.random.seed(42)  # Reproducible

lats = []
lons = []

for _, row in merged.iterrows():
    total = row["num_points"]
    if total <= 0:
        continue

    lat_center = row["lat"]
    lon_center = row["lng"]

    # Layer 1: Super bright core (most dense)
    n_core = int(total * config.CORE_RATIO)
    if n_core > 0:
        lats.append(np.random.normal(lat_center, config.CORE_JITTER, n_core))
        lons.append(np.random.normal(lon_center, config.CORE_JITTER, n_core))

    # Layer 2: Bright area (medium density)
    n_mid = int(total * config.MID_RATIO)
    if n_mid > 0:
        lats.append(np.random.normal(lat_center, config.MID_JITTER, n_mid))
        lons.append(np.random.normal(lon_center, config.MID_JITTER, n_mid))

    # Layer 3: Sparse diffusion (forms gradient edges)
    n_outer = total - n_core - n_mid
    if n_outer > 0:
        lats.append(np.random.normal(lat_center, config.OUTER_JITTER, n_outer))
        lons.append(np.random.normal(lon_center, config.OUTER_JITTER, n_outer))

if not lats:
    raise ValueError("No points generated, please check data or increase POINT_SCALE")

lats = np.concatenate(lats)
lons = np.concatenate(lons)

points = pd.DataFrame({"lat": lats, "lon": lons})

# Downsampling (if too many points)
if len(points) > config.MAX_TOTAL_POINTS:
    points = points.sample(config.MAX_TOTAL_POINTS, random_state=42).reset_index(drop=True)
    print(f"Downsampled to: {len(points):,} points")
else:
    print(f"✓ Final number of points: {len(points):,}")

# ======================
# 6. Load world map
# ======================
print("Loading world map...")

try:
    world_path = geodatasets.get_path("naturalearth.land")
    world = gpd.read_file(world_path)
    print("✓ Map loaded successfully")
except Exception as e:
    print(f"❌ Map loading failed: {e}")
    print("💡 Please install: pip install geodatasets")
    raise

# ======================
# 7. Draw Steam-style map
# ======================
print("Starting to draw...")

fig = plt.figure(figsize=(24, 12), facecolor=config.BG_COLOR, dpi=100)
ax = plt.axes(facecolor=config.BG_COLOR)

# Draw world outline (pure black land)
world.plot(
    ax=ax,
    color=config.LAND_COLOR,
    edgecolor=config.EDGE_COLOR,
    linewidth=0.1
)

# Main point cloud layer
ax.scatter(
    points["lon"],
    points["lat"],
    s=config.POINT_SIZE,
    alpha=config.POINT_ALPHA,
    marker="o",
    linewidths=0,
    c=config.DOT_COLOR,
    rasterized=True  # Accelerate rendering
)

# Glow enhancement layer (optional, makes bright areas brighter)
if config.ENABLE_GLOW_LAYER and len(points) > 1000:
    print(" Adding glow layer...")
    glow_points = points.sample(
        frac=config.GLOW_SAMPLE_RATIO,
        random_state=43
    )
    ax.scatter(
        glow_points["lon"],
        glow_points["lat"],
        s=config.GLOW_SIZE,
        alpha=config.GLOW_ALPHA,
        marker="o",
        linewidths=0,
        c=config.DOT_COLOR,
        rasterized=True
    )

# Set viewport and style
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 85)
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

# Title (Steam style)
title_text = "Global Download Activity - Last 7 Days"
plt.title(
    title_text,
    color="#FFFFFF",
    fontsize=22,
    fontweight="normal",
    pad=25,
    fontfamily="sans-serif"
)

plt.tight_layout()

# ======================
# 8. Save high-resolution image
# ======================
output_file = "steam_official_style_map.png"

plt.savefig(
    output_file,
    dpi=250,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
    edgecolor="none"
)

print(f"Saved successfully: {output_file}")
print(f"Image resolution: {24 * 250} x {12 * 250} px")

plt.show()

# ======================
# 9. Statistics
# ======================
print("\n" + "=" * 50)
print("Statistical Summary")
print("=" * 50)
print(f"Total points: {len(points):,}")
print(f"Cities covered: {len(merged):,}")
print(f"Countries covered: {merged['iso3'].nunique()}")
print(f"Point density coefficient: {config.POINT_SCALE}")
print(f"Core/Mid/Outer ratio: {config.CORE_RATIO}/{config.MID_RATIO}/{config.OUTER_RATIO}")
print("=" * 50)
