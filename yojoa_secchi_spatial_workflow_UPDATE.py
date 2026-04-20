"""
================================================================================
WORKFLOW: Spatially Apply XGBoost Secchi Depth Model Across Lake Yojoa
================================================================================

Based on the 8-step pipeline from yojoa-rs-secchi:
  1. RS collation (Landsat C2 L2 SR from GEE, DSWE water mask, median Rrs)
  2. ERA5 weather summaries (1/3/5/7-day rolling windows)
  3. Regional RS for handoff coefficients
  4. Satellite handoff: polynomial correction LS5/8/9 → LS7
  5. Apply handoff coefficients to Yojoa
  6. Join ERA5 met data with corrected Rrs
  7. Matchup with in-situ secchi + compute band ratios (RN, BG, RB, GB)
  8. XGBoost model (gbtree, reg:squarederror) with 18 features:
     - 4 corrected bands: Blue, Green, Red, Nir
     - 4 band ratios: RN (Red/Nir), BG (Blue/Green), RB (Red/Blue), GB (Green/Blue)
     - 6 ERA5 7-day met: tot_sol_rad, max/mean/min temp, tot_precip, mean_wind
     - 4 ERA5 previous-day met: solar_rad, precip, air_temp, wind_speed

SPATIAL APPLICATION CHALLENGE:
  Your training pipeline uses MEDIAN Rrs across a buffer of DSWE1 (confident
  water) pixels around each sample point. For spatial prediction, we apply the
  model PIXEL-BY-PIXEL, which means each pixel's Rrs values replace the median
  summaries. The ERA5 and handoff corrections are applied uniformly across the
  lake for a given date.

PREREQUISITES:
  pip install earthengine-api geemap xgboost numpy rasterio pandas

  In R, export your chosen model:
    load('data/models/optimized_xg_8_5d_71m.RData')
    xgb.save(optimized_booster_jd_51m, 'xgb_secchi_5d_71m.json')

  Also copy your handoff coefficients CSV alongside this script.
"""

import ee
import geemap
import xgboost as xgb
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import os
import json
from datetime import datetime, timedelta

# ============================================================================
# 0. CONFIGURATION
# ============================================================================

# Lake Yojoa bounding box
LAKE_YOJOA_COORDS = [
    [-88.05, 14.786],
    [-88.05, 14.94],
    [-87.92, 14.94],
    [-87.92, 14.786],
]

# Date range
START_DATE = "2018-01-01"  # LS8 launch — no handoff needed for LS8/9 to LS8/9
END_DATE = "2026-01-01"

# Path to XGBoost model exported from R with xgb.save()
MODEL_PATH = "data/models/xgb_secchi_model_best.json"

# Path to handoff coefficients CSV (from step 4)
HANDOFF_CSV = "data/upstreamRS/yojoa_regional_handoff_coefficients_v2026-03-24.csv"

# The 18 features the model expects, in EXACT order from step 8:
# band_met71_feats in 8_xgboost_stringent.Rmd
FEATURE_NAMES = [
    # 4 corrected Rrs bands
    "med_Blue_corr", "med_Green_corr", "med_Red_corr", "med_Nir_corr",
    # 4 band ratios
    "RN", "BG", "RB", "GB",
    # 7-day ERA5 summaries
    "tot_sol_rad_KJpm2_7", "max_temp_degK_7", "mean_temp_degK_7", "min_temp_degK_7",
    "tot_precip_m_7", "mean_wind_mps_7",
    # Previous-day ERA5
    "solar_rad_KJpm2_prev", "precip_m_prev", "air_temp_degK_prev", "wind_speed_mps_prev",
]

# Landsat band names in GEE (Collection 2 Level 2 Surface Reflectance)
# These map to your med_Blue, med_Green, med_Red, med_Nir, med_Swir1, med_Swir2
GEE_BAND_MAP = {
    "SR_B2": "Blue",   # med_Blue
    "SR_B3": "Green",  # med_Green
    "SR_B4": "Red",    # med_Red
    "SR_B5": "Nir",    # med_Nir
    "SR_B6": "Swir1",  # med_Swir1 (used in DSWE/water masking)
    "SR_B7": "Swir2",  # med_Swir2 (used in DSWE/water masking)
}

OUTPUT_DIR = "secchi_maps_update"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SCALE = 30  # Landsat resolution


# ============================================================================
# 1. INITIALIZE GEE
# ============================================================================

ee.Initialize(
    project="ee-ccmothes",
    opt_url="https://earthengine-highvolume.googleapis.com",
)

lake_aoi = ee.Geometry.Polygon([LAKE_YOJOA_COORDS])


# ============================================================================
# 2. LOAD HANDOFF COEFFICIENTS
#    From step 4/5: polynomial correction LS5/8/9 → LS7 values
#    Format: band, intercept, B1, B2, SatCorr (mission)
#    Correction: corrected = intercept + B1*raw + B2*raw^2
# ============================================================================

def load_handoff_coefficients(csv_path):
    """Load and restructure handoff coefficients from step 4.

    Returns dict like:
      { 'LANDSAT_8': { 'Blue': (intercept, B1, B2), ... }, ... }
    """
    df = pd.read_csv(csv_path)
    if "SatCorr" in df.columns:
        df = df.rename(columns={"SatCorr": "mission"})

    coeffs = {}
    for _, row in df.iterrows():
        mission = row["mission"]
        # band names like "med_Blue" → "Blue"
        band = row["band"].replace("med_", "")
        if mission not in coeffs:
            coeffs[mission] = {}
        coeffs[mission][band] = (row["intercept"], row["B1"], row["B2"])

    return coeffs


def apply_handoff_pixel(value, intercept, b1, b2):
    """Apply polynomial handoff: corrected = intercept + B1*x + B2*x^2"""
    return intercept + b1 * value + b2 * value**2


# ============================================================================
# 3. ERA5 DATA: Fetch 7-day and previous-day summaries from GEE
#    Mirrors step 2: ERA5-Land daily aggregated data
#    For spatial prediction, ERA5 is uniform across the lake (single point)
# ============================================================================

def get_era5_features(date_str):
    """Fetch ERA5 7-day and previous-day weather summaries for a given date.

    Mirrors the rolling-window calculations from 2_Process_Summarize_ERA5.Rmd:
      - 7-day: sum solar rad, mean/min/max temp, sum precip, mean wind
      - Previous day: solar rad, precip, temp, wind speed
      - All summaries use data PRIOR to the observation date (date + 1 offset
        is already applied in the R pipeline)

    Args:
        date_str: 'YYYY-MM-DD' for the Landsat image date

    Returns:
        dict with the 10 ERA5 feature values, or None if data unavailable
    """
    target_date = ee.Date(date_str)
    lake_point = ee.Geometry.Point([-87.875, 14.85])  # Lake Yojoa center

    # ---- Previous day ----
    prev_start = target_date.advance(-1, "day")
    prev_end = target_date

    prev_era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(prev_start, prev_end)
        .first()
    )

    if prev_era5 is None:
        return None

    prev_vals = prev_era5.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=lake_point,
        scale=11132,
    ).getInfo()

    # ---- 7-day window (7 days ending the day before the image) ----
    window_end = target_date
    window_start = target_date.advance(-7, "day")

    era5_7d = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(window_start, window_end)
    )

    n_imgs = era5_7d.size().getInfo()
    if n_imgs < 7:
        print(f"  Warning: only {n_imgs}/7 ERA5 days available for {date_str}")
        if n_imgs == 0:
            return None

    # Solar radiation: SUM over 7 days, convert J/m2 → KJ/m2
    tot_solar = era5_7d.select("surface_solar_radiation_downwards_sum") \
        .sum().reduceRegion(ee.Reducer.mean(), lake_point, 11132).getInfo()

    # Temperature: mean, min, max over 7 days
    temp_mean = era5_7d.select("temperature_2m") \
        .mean().reduceRegion(ee.Reducer.mean(), lake_point, 11132).getInfo()
    temp_min = era5_7d.select("temperature_2m") \
        .min().reduceRegion(ee.Reducer.mean(), lake_point, 11132).getInfo()
    temp_max = era5_7d.select("temperature_2m") \
        .max().reduceRegion(ee.Reducer.mean(), lake_point, 11132).getInfo()

    # Precipitation: SUM over 7 days
    tot_precip = era5_7d.select("total_precipitation_sum") \
        .sum().reduceRegion(ee.Reducer.mean(), lake_point, 11132).getInfo()

    # Wind speed: compute from u,v components per day, then mean
    def add_wind_speed(img):
        u = img.select("u_component_of_wind_10m")
        v = img.select("v_component_of_wind_10m")
        ws = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed")
        return img.addBands(ws)

    mean_wind = era5_7d.map(add_wind_speed).select("wind_speed") \
        .mean().reduceRegion(ee.Reducer.mean(), lake_point, 11132).getInfo()

    # Previous day wind speed from u,v
    prev_u = prev_vals.get("u_component_of_wind_10m", 0) or 0
    prev_v = prev_vals.get("v_component_of_wind_10m", 0) or 0
    prev_wind = np.sqrt(prev_u**2 + prev_v**2)

    return {
        # 7-day summaries
        "tot_sol_rad_KJpm2_7": (tot_solar.get("surface_solar_radiation_downwards_sum", 0) or 0) * 0.001,
        "max_temp_degK_7": temp_max.get("temperature_2m", 0) or 0,
        "mean_temp_degK_7": temp_mean.get("temperature_2m", 0) or 0,
        "min_temp_degK_7": temp_min.get("temperature_2m", 0) or 0,
        "tot_precip_m_7": tot_precip.get("total_precipitation_sum", 0) or 0,
        "mean_wind_mps_7": mean_wind.get("wind_speed", 0) or 0,
        # Previous-day values
        "solar_rad_KJpm2_prev": (prev_vals.get("surface_solar_radiation_downwards_sum", 0) or 0) * 0.001,
        "precip_m_prev": prev_vals.get("total_precipitation_sum", 0) or 0,
        "air_temp_degK_prev": prev_vals.get("temperature_2m", 0) or 0,
        "wind_speed_mps_prev": prev_wind,
    }


# ============================================================================
# 4. LANDSAT PREPROCESSING
#    Mirrors steps 1 and 5:
#    - Cloud/quality masking via QA_PIXEL
#    - DSWE1 water masking (confident water only)
#    - Scale factors applied (C2 L2: multiply 0.0000275, add -0.2)
#    - Rrs range filter: -0.01 < Rrs < 0.2
# ============================================================================

def preprocess_landsat(image):
    """Full Landsat C2 L2 preprocessing for a single image.

    Returns:
        ee.Image with bands: Blue, Green, Red, Nir, Swir1, Swir2
        masked to valid water pixels with Rrs in [-0.01, 0.2]
    """
    # --- Cloud masking via QA_PIXEL (mirrors step 1 filters) ---
    qa = image.select("QA_PIXEL")
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    snow_bit = 1 << 5

    clear_mask = (
        qa.bitwiseAnd(cloud_bit).eq(0)
        .And(qa.bitwiseAnd(shadow_bit).eq(0))
        .And(qa.bitwiseAnd(snow_bit).eq(0))
    )

    # --- Apply C2 L2 scale factors ---
    sr_bands = image.select(list(GEE_BAND_MAP.keys()))
    scaled = sr_bands.multiply(0.0000275).add(-0.2)
    renamed = scaled.rename(list(GEE_BAND_MAP.values()))

    # --- Water masking using NDWI (proxy for DSWE1 confident water) ---
    # Your training uses pCount_dswe1 > 10 from the GEE DSWE algorithm.
    # NDWI > 0 is a reasonable proxy; for exact replication, implement
    # the full DSWE algorithm or use a static lake boundary mask.
    ndwi = renamed.normalizedDifference(["Green", "Nir"]).rename("NDWI")
    water_mask = ndwi.gt(0.0)

    # --- Rrs range filter: keep only [-0.01, 0.2] across all bands ---
    # Mirrors: filter_at(vars(med_Red, ...), all_vars(.<0.2 & . > -0.01))
    bands_to_check = ["Blue", "Green", "Red", "Nir", "Swir1", "Swir2"]
    valid_range = ee.Image.constant(1)
    for band in bands_to_check:
        b = renamed.select(band)
        valid_range = valid_range.And(b.gt(-0.01)).And(b.lt(0.2))

    final_mask = clear_mask.And(water_mask).And(valid_range)
    return renamed.updateMask(final_mask)


# ============================================================================
# 5. PIXEL-LEVEL PREDICTION PIPELINE
# ============================================================================

def predict_secchi_for_date(image_id, model, handoff_coeffs, output_dir=OUTPUT_DIR):
    """Full pipeline for one Landsat scene.

    Steps: preprocess → download → handoff correct → band ratios → ERA5 → predict → GeoTIFF
    """
    image = ee.Image(image_id)
    props = image.getInfo()["properties"]
    mission = props.get("SPACECRAFT_ID", "LANDSAT_8")
    date_str = props.get("DATE_ACQUIRED", "unknown")

    print(f"\nProcessing: {date_str} ({mission})")

    # --- Preprocess (cloud mask, scale, water mask, Rrs filter) ---
    processed = preprocess_landsat(image)

    # --- Download pixel data as numpy ---
    # IMPORTANT: geemap.ee_to_numpy() converts GEE-masked pixels to 0, not NaN.
    # We unmask with a sentinel value (-9999) BEFORE download so masked pixels
    # (clouds, land, out-of-range Rrs) can be identified and set to NaN locally.
    SENTINEL = -9999.0
    band_names = ["Blue", "Green", "Red", "Nir"]

    try:
        np_arr = geemap.ee_to_numpy(
            processed.select(band_names).unmask(SENTINEL),
            region=lake_aoi,
            scale=SCALE,
        )
    except Exception as e:
        print(f"  Skipping: download failed — {e}")
        return None

    if np_arr is None or np_arr.size == 0:
        print(f"  Skipping: no valid pixels")
        return None

    # Convert to float32 and replace sentinel with NaN
    np_arr = np_arr.astype(np.float32)
    np_arr[np_arr == SENTINEL] = np.nan

    h, w, n_bands = np_arr.shape
    print(f"  Image size: {h}x{w}, {n_bands} bands")

    blue = np_arr[:, :, 0]
    green = np_arr[:, :, 1]
    red = np_arr[:, :, 2]
    nir = np_arr[:, :, 3]

    # --- Apply handoff correction (step 5 logic) ---
    if mission in handoff_coeffs:
        coeffs = handoff_coeffs[mission]
        print(f"  Applying {mission} → LS7 handoff correction")
        blue_corr = apply_handoff_pixel(blue, *coeffs["Blue"])
        green_corr = apply_handoff_pixel(green, *coeffs["Green"])
        red_corr = apply_handoff_pixel(red, *coeffs["Red"])
        nir_corr = apply_handoff_pixel(nir, *coeffs["Nir"])
    elif mission == "LANDSAT_7":
        blue_corr, green_corr, red_corr, nir_corr = blue, green, red, nir
    else:
        print(f"  Warning: no handoff coefficients for {mission}, using raw")
        blue_corr, green_corr, red_corr, nir_corr = blue, green, red, nir

    # --- Compute band ratios (from step 7) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        rn = np.where(nir_corr != 0, red_corr / nir_corr, np.nan)
        bg = np.where(green_corr != 0, blue_corr / green_corr, np.nan)
        rb = np.where(blue_corr != 0, red_corr / blue_corr, np.nan)
        gb = np.where(blue_corr != 0, green_corr / blue_corr, np.nan)

    # --- Get ERA5 met features for this date ---
    era5 = get_era5_features(date_str)
    if era5 is None:
        print(f"  Skipping: ERA5 data unavailable for {date_str}")
        return None

    print(f"  ERA5: solar={era5['tot_sol_rad_KJpm2_7']:.0f} KJ/m2, "
          f"temp={era5['mean_temp_degK_7']:.1f} K, "
          f"precip={era5['tot_precip_m_7']:.4f} m")

    # --- Assemble feature array in model's expected order (18 features) ---
    n_pixels = h * w

    feature_array = np.column_stack([
        blue_corr.ravel(),
        green_corr.ravel(),
        red_corr.ravel(),
        nir_corr.ravel(),
        rn.ravel(),
        bg.ravel(),
        rb.ravel(),
        gb.ravel(),
        np.full(n_pixels, era5["tot_sol_rad_KJpm2_7"]),
        np.full(n_pixels, era5["max_temp_degK_7"]),
        np.full(n_pixels, era5["mean_temp_degK_7"]),
        np.full(n_pixels, era5["min_temp_degK_7"]),
        np.full(n_pixels, era5["tot_precip_m_7"]),
        np.full(n_pixels, era5["mean_wind_mps_7"]),
        np.full(n_pixels, era5["solar_rad_KJpm2_prev"]),
        np.full(n_pixels, era5["precip_m_prev"]),
        np.full(n_pixels, era5["air_temp_degK_prev"]),
        np.full(n_pixels, era5["wind_speed_mps_prev"]),
    ])

    # --- Predict with XGBoost ---
    valid_mask = ~np.isnan(feature_array).any(axis=1)
    predictions = np.full(n_pixels, np.nan)

    n_valid = valid_mask.sum()
    if n_valid == 0:
        print(f"  Skipping: no valid pixels after masking")
        return None

    print(f"  Predicting on {n_valid}/{n_pixels} valid water pixels...")
    dmat = xgb.DMatrix(feature_array[valid_mask], feature_names=FEATURE_NAMES)
    predictions[valid_mask] = model.predict(dmat)

    # --- Write GeoTIFF ---
    secchi_map = predictions.reshape(h, w)

    bounds = lake_aoi.bounds().getInfo()["coordinates"][0]
    west = min(c[0] for c in bounds)
    east = max(c[0] for c in bounds)
    south = min(c[1] for c in bounds)
    north = max(c[1] for c in bounds)

    output_path = os.path.join(output_dir, f"secchi_{date_str}_{mission}.tif")
    transform = from_bounds(west, south, east, north, w, h)

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=h, width=w,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(secchi_map.astype("float32"), 1)
        dst.update_tags(
            date=date_str,
            mission=mission,
            model="XGBoost_secchi_5d_71m",
            units="meters",
            features=",".join(FEATURE_NAMES),
        )

    n_predicted = np.isfinite(secchi_map).sum()
    print(f"  Saved: {output_path} ({n_predicted} pixels)")
    return output_path


# ============================================================================
# 6. MAIN WORKFLOW
# ============================================================================

def run_spatial_secchi_workflow():
    """Main entry point: build collection, iterate, predict, export."""

    # Load model
    print("Loading XGBoost model...")
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)
    print(f"  Model loaded: {MODEL_PATH}")

    # Load handoff coefficients
    print("Loading handoff coefficients...")
    handoff_coeffs = load_handoff_coefficients(HANDOFF_CSV)
    for mission, bands in handoff_coeffs.items():
        print(f"  {mission}: {list(bands.keys())}")

    # Build Landsat 8 + 9 collection
    print(f"\nQuerying Landsat imagery: {START_DATE} to {END_DATE}...")

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

    image_ids = collection.aggregate_array("system:index").getInfo()
    print(f"  Raw IDs returned: {len(image_ids)}")
    if image_ids:
        print(f"  First few: {image_ids[:3]}")

    full_ids = []
    for sid in image_ids:
        # Strip any prefix before the LC08/LC09 identifier
        if "LC08" in sid:
            clean = sid[sid.index("LC08"):]
            full_ids.append(f"LANDSAT/LC08/C02/T1_L2/{clean}")
        elif "LC09" in sid:
            clean = sid[sid.index("LC09"):]
            full_ids.append(f"LANDSAT/LC09/C02/T1_L2/{clean}")

    print(f"Found {len(full_ids)} scenes to process")

    output_paths = []
    for i, img_id in enumerate(full_ids):
        print(f"\n{'='*60}")
        print(f"Scene {i+1}/{len(full_ids)}")
        path = predict_secchi_for_date(img_id, bst, handoff_coeffs)
        if path:
            output_paths.append(path)

    print(f"\n{'='*60}")
    print(f"COMPLETE: Generated {len(output_paths)} secchi depth maps")
    print(f"Output directory: {OUTPUT_DIR}")
    return output_paths


# ============================================================================
# 7. R EXPORT HELPER — Run this in R first
# ============================================================================

R_EXPORT_SCRIPT = """
# ============================================================
# Run this in R to export your model for the Python workflow
# ============================================================

library(xgboost)

# Load your best model
load('data/models/optimized_xg_8_5d_71m.RData')

# NOTE: Your script saves optimized_booster_jd_51m into the file named
# "optimized_xg_8_5d_71m.RData". Double-check which booster object is
# your actual best model. The variable name in the RData file may differ
# from what you expect.

# Save as JSON (cross-platform, loads in Python xgboost)
xgb.save(optimized_booster_jd_51m, 'xgb_secchi_5d_71m.json')

# Verify feature names
cat("Feature names in model:\\n")
print(optimized_booster_jd_51m$feature_names)

cat("\\nModel exported. Copy xgb_secchi_5d_71m.json to Python script dir.\\n")
"""


# ============================================================================
# 8. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lake Yojoa Spatial Secchi Depth Prediction Workflow")
    print("Based on yojoa-rs-secchi 8-step pipeline")
    print("=" * 60)
    print()
    print("Before running, ensure you have:")
    print(f"  1. XGBoost model file: {MODEL_PATH}")
    print(f"  2. Handoff coefficients: {HANDOFF_CSV}")
    print(f"  3. GEE authentication (ee.Authenticate())")
    print()
    print("To export model from R:")
    print(R_EXPORT_SCRIPT)

    # Uncomment to run:
    run_spatial_secchi_workflow()