import streamlit as st
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import requests
import tempfile
from rasterio.transform import xy
import matplotlib.pyplot as plt

st.set_page_config(page_title="Raster Distribution Viewer", layout="wide")

st.title("Raster Distribution Analysis by Country")

# === User Inputs ===
uploaded_file = st.file_uploader("Upload GeoTIFF file", type=["tif"])
value_min = st.slider("Minimum Value", 0.0, 100.0, 95.0)
value_max = st.slider("Maximum Value", 0.0, 100.0, 100.0)
top_n = st.number_input("Top N Countries to Display", min_value=1, value=10)

if uploaded_file:
    # === Load Country Boundaries ===
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
            pixel_area_km2 = (transform[0] * -transform[4]) / 1e6
            rows, cols = np.where(match_mask)
            xs, ys = xy(transform, rows, cols)
            points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=crs)

            if points.crs != world.crs:
                points = points.to_crs(world.crs)

            joined = gpd.sjoin(points, world, how="inner", predicate="intersects")
            summary = joined.groupby("ADMIN").size().reset_index(name="matched_pixels")
            summary["area_km2"] = summary["matched_pixels"] * pixel_area_km2
            summary["area_ha"] = summary["area_km2"] * 100
            world["country_area_km2"] = world.geometry.area / 1e6
            summary = summary.merge(world[["ADMIN", "country_area_km2"]], on="ADMIN", how="left")
            summary["percent_covered"] = 100 * summary["area_km2"] / summary["country_area_km2"]

            st.subheader("Country Summary")
            st.dataframe(summary.sort_values("matched_pixels", ascending=False))

            top_summary = summary.sort_values("matched_pixels", ascending=False).head(top_n)
            fig2, ax2 = plt.subplots()
            ax2.bar(top_summary["ADMIN"], top_summary["matched_pixels"], color="tomato", edgecolor="black")
            ax2.set_title(f"Top {top_n} Countries by Matched Pixels")
            ax2.set_ylabel("Matched Pixels")
            ax2.set_xlabel("Country")
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)

            top_covered = summary.sort_values("percent_covered", ascending=False).head(top_n)
            fig3, ax3 = plt.subplots()
            ax3.bar(top_covered["ADMIN"], top_covered["percent_covered"], color="seagreen", edgecolor="black")
            ax3.set_title(f"Top {top_n} Countries by % Covered")
            ax3.set_ylabel("Percent Covered (%)")
            ax3.set_xlabel("Country")
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)
