"""
================================================================================
WORKFLOW: Export True-Color RGB Landsat GeoTIFFs for Lake Yojoa
================================================================================

Companion to yojoa_secchi_spatial_workflow.py. Exports one RGB GeoTIFF per
Landsat scene that matches a secchi_<date>_<mission>.tif output, enabling the
Shiny app to toggle between true-color imagery and modeled Secchi depth.

Output filename convention (matches secchi workflow):
    rgb_maps/rgb_<date>_<mission>.tif   e.g. rgb_2023-03-14_LANDSAT_8.tif
    secchi_maps/secchi_<date>_<mission>.tif  (from secchi workflow)

The RGB raster:
  - 3-band GeoTIFF (R, G, B), uint8 (0–255), same extent and CRS as secchi TIFs
  - Uses the same QA_PIXEL cloud mask as the secchi preprocessing
  - Water mask is NOT applied — you want to see the full scene incl. land context
  - Bands: SR_B4 (Red), SR_B3 (Green), SR_B2 (Blue) from C2 L2
  - Stretch: linear 2%–98% per-scene stretch → uint8, with optional fixed stretch

PREREQUISITES:
    Same environment as secchi workflow:
    pip install earthengine-api geemap rasterio numpy pandas
"""

import ee
import geemap
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os
import pandas as pd
from pathlib import Path

# ============================================================================
# 0. CONFIGURATION — keep in sync with secchi workflow
# ============================================================================

LAKE_YOJOA_COORDS = [
    [-88.05, 14.786],
    [-88.05, 14.94],
    [-87.92, 14.94],
    [-87.92, 14.786],
]

START_DATE = "2018-01-01"
END_DATE   = "2026-01-01"

OUTPUT_DIR = "rgb_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALE = 30  # Match secchi workflow resolution

# Stretch mode:
#   "per_scene"  — linear 2%-98% stretch computed from each scene's valid pixels
#                  (best visual quality, but brightness varies between scenes)
#   "fixed"      — apply a fixed Rrs range common for tropical lakes
#                  (consistent brightness across scenes; better for time-series)
STRETCH_MODE = "per_scene"

# Used only when STRETCH_MODE = "fixed". Rrs values mapped to 0 and 255.
# Typical C2 L2 surface reflectance range over water: ~0.0 – 0.12
FIXED_MIN = 0.0
FIXED_MAX = 0.12

# Set to a path of existing secchi TIFs to only process matching scenes,
# or None to process all scenes in the date range.
SECCHI_DIR = "secchi_maps"   # set to None to ignore


# ============================================================================
# 1. INITIALIZE GEE
# ============================================================================

ee.Initialize(
    project="ee-ccmothes",
    opt_url="https://earthengine-highvolume.googleapis.com",
)

lake_aoi = ee.Geometry.Polygon([LAKE_YOJOA_COORDS])


# ============================================================================
# 2. CLOUD-MASK AND SCALE LANDSAT FOR TRUE COLOR
#    Same QA_PIXEL masking as secchi workflow (cloud, shadow, snow bits).
#    Water mask intentionally omitted — land context aids visual interpretation.
# ============================================================================

def preprocess_rgb(image):
    """Apply cloud mask and C2 L2 scale factors; return Red, Green, Blue bands."""
    qa = image.select("QA_PIXEL")
    cloud_bit  = 1 << 3
    shadow_bit = 1 << 4
    snow_bit   = 1 << 5

    clear_mask = (
        qa.bitwiseAnd(cloud_bit).eq(0)
        .And(qa.bitwiseAnd(shadow_bit).eq(0))
        .And(qa.bitwiseAnd(snow_bit).eq(0))
    )

    # C2 L2 scale: DN * 0.0000275 - 0.2  → Rrs
    rgb = image.select(["SR_B4", "SR_B3", "SR_B2"]).multiply(0.0000275).add(-0.2)
    rgb = rgb.rename(["Red", "Green", "Blue"])
    return rgb.updateMask(clear_mask)


# ============================================================================
# 3. STRETCH HELPERS: Rrs → uint8
# ============================================================================

def stretch_to_uint8(arr_float, lo=None, hi=None, pct_low=2, pct_high=98):
    """Linear stretch of a float32 Rrs array to uint8 (0–255).

    If lo/hi not provided, uses percentile stretch over valid (non-NaN) pixels.
    NaN pixels → 0 in output (transparent/nodata in the TIF).
    """
    valid = arr_float[np.isfinite(arr_float)]
    if valid.size == 0:
        return np.zeros_like(arr_float, dtype=np.uint8)

    if lo is None:
        lo = np.percentile(valid, pct_low)
    if hi is None:
        hi = np.percentile(valid, pct_high)

    if hi == lo:
        return np.zeros_like(arr_float, dtype=np.uint8)

    clipped = np.clip(arr_float, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    result = scaled.astype(np.uint8)
    result[~np.isfinite(arr_float)] = 0
    return result


# ============================================================================
# 4. SINGLE-SCENE RGB EXPORT
# ============================================================================

def export_rgb_for_scene(image_id, output_dir=OUTPUT_DIR):
    """Download and write a true-color RGB GeoTIFF for one Landsat scene.

    Args:
        image_id: Full GEE asset path, e.g. 'LANDSAT/LC08/C02/T1_L2/LC08_...'
        output_dir: Where to write the GeoTIFF

    Returns:
        Path to output file, or None if skipped.
    """
    image = ee.Image(image_id)
    props  = image.getInfo()["properties"]
    mission  = props.get("SPACECRAFT_ID", "LANDSAT_8")
    date_str = props.get("DATE_ACQUIRED", "unknown")

    out_path = os.path.join(output_dir, f"rgb_{date_str}_{mission}.tif")

    if os.path.exists(out_path):
        print(f"  Already exists, skipping: {out_path}")
        return out_path

    print(f"\nProcessing RGB: {date_str} ({mission})")

    processed = preprocess_rgb(image)

    try:
        np_arr = geemap.ee_to_numpy(
            processed.select(["Red", "Green", "Blue"]),
            region=lake_aoi,
            scale=SCALE,
        )
    except Exception as e:
        print(f"  Skipping: download failed — {e}")
        return None

    if np_arr is None or np_arr.size == 0:
        print(f"  Skipping: no valid pixels returned")
        return None

    h, w, _ = np_arr.shape
    print(f"  Image size: {h}x{w}")

    red_f   = np_arr[:, :, 0].astype(np.float32)
    green_f = np_arr[:, :, 1].astype(np.float32)
    blue_f  = np_arr[:, :, 2].astype(np.float32)

    # Replace 0.0 fill values from geemap with NaN (masked pixels come back as 0)
    # Only affects pixels where ALL bands are exactly 0 (true water pixels are >0)
    # A safer mask: any value ≤ -0.01 (physically impossible Rrs) → NaN
    for arr in [red_f, green_f, blue_f]:
        arr[arr < -0.01] = np.nan

    if STRETCH_MODE == "fixed":
        red_u8   = stretch_to_uint8(red_f,   lo=FIXED_MIN, hi=FIXED_MAX)
        green_u8 = stretch_to_uint8(green_f, lo=FIXED_MIN, hi=FIXED_MAX)
        blue_u8  = stretch_to_uint8(blue_f,  lo=FIXED_MIN, hi=FIXED_MAX)
    else:
        # Per-scene: compute a single lo/hi from all valid pixels across all 3 bands
        # (preserves relative band brightness, avoids per-band color distortion)
        all_valid = np.concatenate([
            red_f[np.isfinite(red_f)],
            green_f[np.isfinite(green_f)],
            blue_f[np.isfinite(blue_f)],
        ])
        if all_valid.size == 0:
            print(f"  Skipping: no valid pixels after masking")
            return None
        lo = np.percentile(all_valid, 2)
        hi = np.percentile(all_valid, 98)
        red_u8   = stretch_to_uint8(red_f,   lo=lo, hi=hi)
        green_u8 = stretch_to_uint8(green_f, lo=lo, hi=hi)
        blue_u8  = stretch_to_uint8(blue_f,  lo=lo, hi=hi)

    # Build nodata mask: pixels where ANY band was NaN → alpha=0
    nodata_mask = ~(np.isfinite(np_arr[:, :, 0]) &
                    np.isfinite(np_arr[:, :, 1]) &
                    np.isfinite(np_arr[:, :, 2]))

    # Get bounds from AOI
    bounds = lake_aoi.bounds().getInfo()["coordinates"][0]
    west  = min(c[0] for c in bounds)
    east  = max(c[0] for c in bounds)
    south = min(c[1] for c in bounds)
    north = max(c[1] for c in bounds)
    transform = from_bounds(west, south, east, north, w, h)

    # Write 4-band GeoTIFF: R, G, B, Alpha (255=valid, 0=nodata/cloud/land-if-masked)
    alpha = np.where(nodata_mask, np.uint8(0), np.uint8(255))

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=h, width=w,
        count=4,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(red_u8,   1)
        dst.write(green_u8, 2)
        dst.write(blue_u8,  3)
        dst.write(alpha,    4)
        dst.update_tags(
            date=date_str,
            mission=mission,
            stretch=STRETCH_MODE,
            stretch_lo=str(lo if STRETCH_MODE == "per_scene" else FIXED_MIN),
            stretch_hi=str(hi if STRETCH_MODE == "per_scene" else FIXED_MAX),
        )

    print(f"  Saved: {out_path}")
    return out_path


# ============================================================================
# 5. MAIN WORKFLOW
# ============================================================================

def run_rgb_export_workflow():
    """Export RGB GeoTIFFs for all (or matching) Landsat scenes."""

    # ---- Build scene list ----
    if SECCHI_DIR and os.path.isdir(SECCHI_DIR):
        # Match only scenes that already have a secchi TIF (recommended)
        secchi_files = list(Path(SECCHI_DIR).glob("secchi_*.tif"))
        print(f"Found {len(secchi_files)} existing secchi TIFs — exporting matching RGB scenes")

        full_ids = []
        for sf in sorted(secchi_files):
            # secchi_2023-03-14_LANDSAT_8.tif → date=2023-03-14, mission=LANDSAT_8
            stem = sf.stem  # "secchi_2023-03-14_LANDSAT_8"
            parts = stem.split("_", 1)[1]               # "2023-03-14_LANDSAT_8"
            date_part, mission_part = parts.rsplit("_", 2)[0], "_".join(parts.rsplit("_")[1:])
            # mission_part is like "LANDSAT_8" → satellite = LC08 or LC09
            sat = "LC08" if "8" in mission_part else "LC09"
            col = "LANDSAT/LC08/C02/T1_L2" if sat == "LC08" else "LANDSAT/LC09/C02/T1_L2"
            # Query GEE for the scene on that date
            date_ee = ee.Date(date_part)
            img = (
                ee.ImageCollection(col)
                .filterBounds(lake_aoi)
                .filterDate(date_ee, date_ee.advance(1, "day"))
                .first()
            )
            # Get the actual image ID
            try:
                sid = img.get("system:index").getInfo()
                if sid:
                    clean = sid[sid.index(sat):]
                    full_ids.append(f"{col}/{clean}")
            except Exception as e:
                print(f"  Could not resolve GEE ID for {date_part}: {e}")

    else:
        # No secchi dir — process all scenes in date range
        print(f"Querying all Landsat scenes: {START_DATE} to {END_DATE}...")
        ls8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(lake_aoi)
            .filterDate(START_DATE, END_DATE)
            .filter(ee.Filter.lt("CLOUD_COVER", 50))
        )
        ls9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(lake_aoi)
            .filterDate(START_DATE, END_DATE)
            .filter(ee.Filter.lt("CLOUD_COVER", 50))
        )
        collection = ls8.merge(ls9).sort("DATE_ACQUIRED")
        image_ids = collection.aggregate_array("system:index").getInfo()

        full_ids = []
        for sid in image_ids:
            if "LC08" in sid:
                clean = sid[sid.index("LC08"):]
                full_ids.append(f"LANDSAT/LC08/C02/T1_L2/{clean}")
            elif "LC09" in sid:
                clean = sid[sid.index("LC09"):]
                full_ids.append(f"LANDSAT/LC09/C02/T1_L2/{clean}")

    print(f"\nScenes to process: {len(full_ids)}")

    output_paths = []
    for i, img_id in enumerate(full_ids):
        print(f"\n{'='*60}")
        print(f"Scene {i+1}/{len(full_ids)}: {img_id}")
        path = export_rgb_for_scene(img_id)
        if path:
            output_paths.append(path)

    print(f"\n{'='*60}")
    print(f"COMPLETE: Generated {len(output_paths)} RGB GeoTIFFs → {OUTPUT_DIR}/")
    return output_paths


# ============================================================================
# 6. SHINY INTEGRATION NOTES
# ============================================================================

SHINY_NOTES = """
R / Shiny integration
---------------------

Both raster types share the same filename stem after the prefix:
    secchi_maps/secchi_2023-03-14_LANDSAT_8.tif
    rgb_maps/rgb_2023-03-14_LANDSAT_8.tif

To load in R:
    library(terra)

    # Secchi (single-band float)
    r_secchi <- rast("secchi_maps/secchi_2023-03-14_LANDSAT_8.tif")

    # RGB (4-band uint8: R, G, B, Alpha)
    r_rgb <- rast("rgb_maps/rgb_2023-03-14_LANDSAT_8.tif")

In leaflet with leafem:
    # True-color — use leafem::addRasterRGB (bands 1,2,3)
    leaflet() |>
      addTiles() |>
      addRasterRGB(r_rgb, r = 1, g = 2, b = 3, maxBytes = 8 * 1024 * 1024)

    # Secchi depth — use addRasterImage with a color palette
    pal <- colorNumeric("viridis", values(r_secchi), na.color = "transparent")
    leaflet() |>
      addTiles() |>
      addRasterImage(r_secchi, colors = pal, opacity = 0.85)

UI toggle (bslib / shiny):
    radioButtons("layer_type", "View:",
                 choices = c("Secchi Depth" = "secchi", "True Color" = "rgb"),
                 inline = TRUE)

    # In server: swap which raster is rendered based on input$layer_type
"""

# ============================================================================
# 7. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lake Yojoa RGB True-Color Export Workflow")
    print("Companion to yojoa_secchi_spatial_workflow.py")
    print("=" * 60)
    print()
    print(f"Stretch mode : {STRETCH_MODE}")
    print(f"Output dir   : {OUTPUT_DIR}/")
    print(f"Match secchi : {SECCHI_DIR or 'No — processing all scenes'}")
    print()
    print(SHINY_NOTES)

    # Uncomment to run:
    run_rgb_export_workflow()