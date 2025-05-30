from __future__ import annotations
from typing import Union, Dict, Any

from shapely.geometry import shape, Polygon, MultiPolygon
import geopandas as gpd
import pandas as pd
from OSMPythonTools.overpass import Overpass

def _geojson_to_polygon(aoi: Union[Dict[str, Any], Polygon, MultiPolygon]) -> Polygon | MultiPolygon:
    """Accept a GeoJSON dict *or* Shapely geometry and return a Shapely polygon/multipolygon."""
    if isinstance(aoi, (Polygon, MultiPolygon)):
        return aoi
    if isinstance(aoi, dict):
        return shape(aoi)
    raise TypeError("AOI must be a GeoJSON-like dict or a shapely Polygon/MultiPolygon")

def _poly_to_overpass_string(poly: Polygon | MultiPolygon) -> str:
    """Convert exterior ring(s) to the 'lat lon lat lon …' format Overpass expects."""
    if isinstance(poly, Polygon):
        rings = [poly.exterior.coords]
    else:  # MultiPolygon
        rings = [p.exterior.coords for p in poly.geoms]

    parts = []
    for ring in rings:
        parts.extend(f"{lat} {lon}" for lon, lat in ring)
    return " ".join(parts)

def _build_query(coords_str: str) -> str:
    """Raw Overpass QL requesting full geometry for land-use / natural / land-cover."""
    return f"""
    (
      way["landuse"](poly:"{coords_str}");
      way["natural"](poly:"{coords_str}");
      way["landcover"](poly:"{coords_str}");
      relation["landuse"](poly:"{coords_str}");
      relation["natural"](poly:"{coords_str}");
      relation["landcover"](poly:"{coords_str}");
    );
    out geom;
    """

def _elements_to_gdf(result) -> gpd.GeoDataFrame:
    """Convert OSMPythonTools elements → GeoPandas GeoDataFrame (EPSG:4326)."""
    geometries, attrs = [], []
    for el in result.elements():
        tags = el.tags()
        if not tags:
            continue
        geom = shape(el.geometry())
        if not geom.geom_type.startswith("Polygon"):
            continue
        geometries.append(geom)
        attrs.append(tags)

    gdf = gpd.GeoDataFrame(attrs, geometry=geometries, crs="EPSG:4326")
    # gdf["cover"] = (
    #     gdf.get("landuse")
    #        .fillna(gdf.get("natural"))
    #        # .fillna(gdf.get("landcover"))
    # )
    cols = [c for c in ("landuse", "natural", "landcover") if c in gdf.columns]
    if cols:                                      # at least one tag present
        gdf["cover"] = gdf[cols].bfill(axis=1).iloc[:, 0]
    else:                                         # none of the tags exist
        gdf["cover"] = pd.Series(pd.NA, index=gdf.index, dtype="object")

    return gdf

# ---------------------------------------------------------------------------

def get_osm_landcover(
    aoi_geojson: Union[Dict[str, Any], Polygon, MultiPolygon],
    out_crs: str | int = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """
    Retrieve OSM land-use / natural / land-cover polygons intersecting *aoi_geojson*.

    Parameters
    ----------
    aoi_geojson : dict | shapely Polygon/MultiPolygon
        The area of interest.  If dict, must be GeoJSON-like.
    out_crs : str | int, default "EPSG:3857"
        The CRS you want the result in (e.g. 4326, 3857, or any proj string).

    Returns
    -------
    GeoDataFrame
        Columns: all original OSM tag columns + 'cover' (categorical label).
        CRS: *out_crs*.
    """
    # 1. normalise AOI input
    poly = _geojson_to_polygon(aoi_geojson)

    # 2. build & run Overpass query
    overpass = Overpass()
    query = _build_query(_poly_to_overpass_string(poly))
    result = overpass.query(query)

    # 3. convert to GeoDataFrame
    gdf = _elements_to_gdf(result)

    # 4. clip to AOI
    aoi_gs = gpd.GeoSeries([poly], crs="EPSG:4326")
    gdf = gpd.clip(gdf, aoi_gs)

    # 5. project to requested CRS
    if out_crs:
        gdf = gdf.to_crs(out_crs)

    return gdf
