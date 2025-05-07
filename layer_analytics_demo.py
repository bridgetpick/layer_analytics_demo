import streamlit as st
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import requests
import tempfile
from rasterio.transform import xy
from shapely.geometry import box
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Raster Distribution Viewer", layout="wide")
st.title("Raster Distribution Analysis by Country")

# === USER INPUTS ===

# Directory containing your GeoTIFF files (must be relative to this script's location)
tif_dir = "tif_directory"  # Make sure this folder exists and is at the same level as this script
if not os.path.isdir(tif_dir):
    st.error(f"Directory '{tif_dir}' not found. Please check the path.")
    st.stop()

# Dropdown menu to select a .tif file
available_tifs = [f for f in os.listdir(tif_dir) if f.endswith(".tif")]
if not available_tifs:
    st.warning("No .tif files found in the directory.")
    st.stop()

selected_filename = st.selectbox("Select a GeoTIFF file", available_tifs)
uploaded_file = os.path.join(tif_dir, selected_filename)
st.success(f"Loaded: {selected_filename}")

# Other controls
value_min = st.slider("Minimum Value", 0.0, 100.0, 95.0)
value_max = st.slider("Maximum Value", 0.0, 100.0, 100.0)
top_n = st.number_input("Top N Countries to Display", min_value=1, value=10)

def clean_label(label, max_chars=12):
    return '\n'.join([label[i:i+max_chars] for i in range(0, len(label), max_chars)])

# === MAIN PROCESSING ===
if uploaded_file:
    # Load world boundaries
    url = "https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/ne_110m_admin_0_countries.geojson"
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmpfile:
        tmpfile.write(response.content)
        shapefile_path = tmpfile.name
    world = gpd.read_file(shapefile_path)

    with rasterio.open(uploaded_file) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        band = np.where(band == nodata, np.nan, band)
        band_flat = band[~np.isnan(band)]

        # Histogram
        st.subheader("Histogram of All Raster Values")
        fig1, ax1 = plt.subplots()
        ax1.hist(band_flat, bins=50, color='steelblue', edgecolor='black')
        ax1.set_title("Raster Value Distribution")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        match_mask = (band >= value_min) & (band <= value_max)
        matched_pixels = np.sum(match_mask)
        st.write(f"**Matched Pixels:** {matched_pixels}")

        if matched_pixels == 0:
            st.warning("No pixels found in the specified range.")
        else:
            rows, cols = np.where(match_mask)
            xs, ys = xy(transform, rows, cols)

            # Build square polygons for each matched pixel
            pixel_width = transform[0]
            pixel_height = -transform[4]
            polygons = [
                box(x - pixel_width / 2, y - pixel_height / 2,
                    x + pixel_width / 2, y + pixel_height / 2)
                for x, y in zip(xs, ys)
            ]
            pixels = gpd.GeoDataFrame(geometry=polygons, crs=crs)

            # Reproject to equal-area CRS
            equal_area_crs = "EPSG:6933"
            pixels = pixels.to_crs(equal_area_crs)
            world_proj = world.to_crs(equal_area_crs)

            # Calculate pixel area
            pixels["pixel_area_km2"] = pixels.geometry.area / 1e6

            # Spatial join
            joined = gpd.sjoin(pixels, world_proj, how="inner", predicate="intersects")

            # Aggregate by country
            summary = joined.groupby("ADMIN").agg(
                matched_pixels=("geometry", "count"),
                area_km2=("pixel_area_km2", "sum")
            ).reset_index()

            # Country area and coverage
            world_proj["country_area_km2"] = world_proj.geometry.area / 1e6
            summary = summary.merge(world_proj[["ADMIN", "country_area_km2"]], on="ADMIN", how="left")
            summary["percent_covered"] = (100 * summary["area_km2"] / summary["country_area_km2"]).round(4)
            summary["area_km2"] = summary["area_km2"].round(2)

            # Display table
            display_summary = summary[["ADMIN", "matched_pixels", "area_km2", "percent_covered"]]
            st.subheader("Country Summary Table")
            st.dataframe(display_summary.sort_values("matched_pixels", ascending=False))

            # Bar plot: matched pixels
            top_summary = display_summary.sort_values("matched_pixels", ascending=False).head(top_n)
            top_summary["ADMIN"] = top_summary["ADMIN"].apply(clean_label)

            st.subheader(f"Top {top_n} Countries by Matched Pixels")
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            ax2.bar(top_summary["ADMIN"], top_summary["matched_pixels"], color="tomato", edgecolor="black")
            ax2.set_ylabel("Matched Pixels")
            ax2.set_xlabel("Country")
            ax2.set_title("Matched Pixel Count")
            ax2.tick_params(axis='x', labelsize=10)
            plt.xticks(rotation=30, ha="center")
            plt.tight_layout()
            st.pyplot(fig2)

            # Bar plot: percent covered
            top_covered = display_summary.sort_values("percent_covered", ascending=False).head(top_n)
            top_covered["ADMIN"] = top_covered["ADMIN"].apply(clean_label)

            st.subheader(f"Top {top_n} Countries by Percent of Area Covered")
            fig3, ax3 = plt.subplots(figsize=(14, 6))
            ax3.bar(top_covered["ADMIN"], top_covered["percent_covered"], color="seagreen", edgecolor="black")
            ax3.set_ylabel("Percent Covered (%)")
            ax3.set_xlabel("Country")
            ax3.set_title("Country Area % Covered by Matched Pixels")
            ax3.tick_params(axis='x', labelsize=10)
            plt.xticks(rotation=30, ha="center")
            plt.tight_layout()
            st.pyplot(fig3)
