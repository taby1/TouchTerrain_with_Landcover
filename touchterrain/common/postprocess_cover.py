# postprocess_cover.py  ────────────────────────────────────────────────────
from __future__ import annotations

from typing import List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# -------------------------------------------------------------------------
def remove_overlaps(
    gdf: gpd.GeoDataFrame,
    cover_col: str = "cover",
) -> gpd.GeoDataFrame:
    """
    Resolve overlapping polygons by **keeping the larger area** feature.

    Any overlap is subtracted from smaller polygons so the result layer has
    no geometry intersections at all.

    Returns
    -------
    GeoDataFrame without overlaps.
    """
    # sort by area descending
    gdf_tmp = gdf.assign(_area=gdf.area).sort_values("_area", ascending=False)
    kept_rows: List[dict] = []
    union_kept = None

    for _, row in gdf_tmp.iterrows():
        geom = row.geometry
        if union_kept:
            geom = geom.difference(union_kept)
        if geom.is_empty:
            continue

        new_row = row.drop(labels=["_area"]).to_dict()
        new_row["geometry"] = geom
        kept_rows.append(new_row)

        union_kept = union_kept.union(geom) if union_kept else geom

    return gpd.GeoDataFrame(kept_rows, crs=gdf.crs).reset_index(drop=True)


# -------------------------------------------------------------------------
def smooth_cover_layer(
    gdf: gpd.GeoDataFrame,
    cover_col: str = "cover",
    tol: float = 10.0,
    min_area: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Morphological smoothing: erode by *tol* then dilate by *tol*.
    Removes corridors/spikes narrower than 2×tol.  Parts < *min_area* drop.
    """
    if min_area is None:
        min_area = tol ** 2

    new_geom, new_cov = [], []

    for cov, geom in zip(gdf[cover_col], gdf.geometry):
        try:
            g = geom.buffer(-tol).buffer(tol)
        except ValueError:
            continue
        if g.is_empty:
            continue

        parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
        for p in parts:
            if p.area >= min_area:
                new_geom.append(p)
                new_cov.append(cov)

    if not new_geom:
        return gdf.copy()

    smoothed = gpd.GeoDataFrame(
        {cover_col: new_cov, "geometry": new_geom}, crs=gdf.crs
    )
    return (
        smoothed
        .dissolve(by=cover_col, as_index=False)
        .explode(index_parts=False)
        .reset_index(drop=True)
    )


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
#  REPLACE OLD expand_to_fill_aoi WITH THIS ITERATIVE VERSION
# -------------------------------------------------------------------------
def expand_to_fill_aoi(
    gdf: gpd.GeoDataFrame,
    aoi: Union[Polygon, MultiPolygon, gpd.GeoSeries],
    cover_col: str = "cover",
    pixel_size: float = 5.0,        # metres per pixel in output grid
) -> gpd.GeoDataFrame:
    """
    Fill every remaining gap by assigning each map pixel to the **closest
    polygon boundary** (Euclidean distance).

    Parameters
    ----------
    pixel_size : float
        Output raster resolution.  Use ~ map-scale pixel size; e.g. 5 m
        makes final slivers < 5 m vanish.

    Returns
    -------
    GeoDataFrame  – polygons after simultaneous expansion.
    """
    # 0.  Geometry prep ---------------------------------------------------
    if isinstance(aoi, gpd.GeoSeries):
        aoi_poly = aoi.iloc[0]
    else:
        aoi_poly = aoi

    # Ensure the layer has an integer ID for raster labels
    gdf = gdf.reset_index(drop=True).copy()
    gdf["_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)

    # 1.  Build an “analysis raster” covering the AOI ---------------------
    minx, miny, maxx, maxy = aoi_poly.bounds
    width  = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))

    transform = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)
    out_shape = (height, width)

    # 1a. Rasterise ONLY the polygon interiors (id value), 0 = nodata
    interior = features.rasterize(
        ((geom, fid) for geom, fid in zip(gdf.geometry, gdf["_id"])),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,          # only interior pixels
    )

    # 1b. Mask exterior-of-AOI pixels so we never assign outside area
    aoi_mask = features.geometry_mask(
        [aoi_poly], out_shape=out_shape, transform=transform, invert=True
    )
    interior = np.where(aoi_mask, interior, -1)     # -1 = outside study area

    # 2.  Distance transform on the “gaps” (pixels == 0) ------------------
    gap_mask = interior == 0
    if not gap_mask.any():
        # already full – just vectorise & quit
        return gdf.drop(columns="_id")

    # Compute distance to nearest labelled pixel & the argmin label
    # Step 1: distance to *all* non-gap pixels
    dist, indices = ndi.distance_transform_edt(
        gap_mask,
        return_indices=True,
    )

    # Map row/col of nearest source pixel back to its label
    nearest_labels = interior[indices[0], indices[1]]
    # Write these labels into the gap pixels
    filled = interior.copy()
    filled[gap_mask] = nearest_labels[gap_mask]

    # 3.  Vectorise back to polygons --------------------------------------
    shapes = features.shapes(filled.astype("int32"), mask=(filled > 0), transform=transform)
    records = []
    for geom, value in shapes:
        poly = shape(geom)
        fid  = int(value)
        if poly.is_empty:
            continue
        records.append({"_id": fid, "geometry": poly})

    expanded = gpd.GeoDataFrame(records, crs=gdf.crs)

    # 4.  Join cover attribute back, dissolve on cover --------------------
    expanded = expanded.merge(
        gdf[["_id", cover_col]], on="_id", how="left", validate="m:1"
    ).drop(columns="_id")

    expanded = (
        expanded.dissolve(by=cover_col, as_index=False)
                .explode(index_parts=False)
                .reset_index(drop=True)
    )
    return expanded

# -------------------------------------------------------------------------
#  FINAL GAP FILL – push polygons out to the AOI boundary
# -------------------------------------------------------------------------
def fill_to_aoi_boundary(
    gdf: gpd.GeoDataFrame,
    aoi: Union[Polygon, MultiPolygon, gpd.GeoSeries],
    cover_col: str = "cover",
    max_iter: int = 3,            # usually one pass is enough
) -> gpd.GeoDataFrame:
    """
    Close any residual gaps (typically slender strips along the AOI edge)
    by assigning each gap the *cover* of its neighbouring polygon that has
    the longest shared boundary with it.

    The process repeats up to *max_iter* times; it almost always converges
    after the first pass once the big raster-based expansion has run.
    """
    if isinstance(aoi, gpd.GeoSeries):
        aoi_poly = aoi.iloc[0]
    else:
        aoi_poly = aoi

    work = gdf.copy()

    for _ in range(max_iter):
        gap_poly = aoi_poly.difference(unary_union(work.geometry))
        if gap_poly.is_empty:
            break

        gaps = list(gap_poly.geoms) if gap_poly.geom_type == "MultiPolygon" else [gap_poly]
        additions = []

        for gap in gaps:
            # neighbours that touch this gap
            nbrs = work[work.touches(gap)]
            if nbrs.empty:
                # this can happen if the gap only touches AOI edge
                continue

            shared_len = nbrs.geometry.boundary.intersection(gap.boundary).length
            best_idx   = shared_len.idxmax()
            best_cover = nbrs.loc[best_idx, cover_col]

            additions.append({cover_col: best_cover, "geometry": gap})

        if not additions:
            break

        work = pd.concat(
            [work, gpd.GeoDataFrame(additions, crs=work.crs)],
            ignore_index=True,
        )
        # dissolve so the next round sees merged polygons
        work = (
            work.dissolve(by=cover_col, as_index=False)
                .explode(index_parts=False)
                .reset_index(drop=True)
        )

    return work

# -------------------------------------------------------------------------
def dissolve_cover(
    gdf: gpd.GeoDataFrame,
    cover_col: str = "cover",
) -> gpd.GeoDataFrame:
    """Merge adjacent polygons that share the same cover value."""
    return (
        gdf.dissolve(by=cover_col, as_index=False)
           .explode(index_parts=False)
           .reset_index(drop=True)
    )


# -------------------------------------------------------------------------
def refine_cover_layer(
    gdf, aoi,
    cover_col="cover",
    smooth_tol=10.0,
    pixel_size=5.0,
):
    work = remove_overlaps(gdf, cover_col=cover_col)
    work = dissolve_cover(work, cover_col=cover_col)
    work = smooth_cover_layer(work, cover_col=cover_col, tol=smooth_tol)
    work = expand_to_fill_aoi(work, aoi,
                              cover_col=cover_col,
                              pixel_size=pixel_size)
    work = fill_to_aoi_boundary(work, aoi, cover_col=cover_col)
    # tidy
    work = dissolve_cover(work, cover_col=cover_col)
    return work.reset_index(drop=True)
