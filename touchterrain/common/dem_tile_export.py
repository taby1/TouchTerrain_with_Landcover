"""
DEM Tile Export Pipeline
-------------------------

This module provides a complete workflow for extracting a DEM from Earth Engine,
converting it to a triangle mesh, constructing a watertight block, clipping it to a
GeoJSON AOI polygon, and exporting as an STL file.

Steps:
    1. fetch_dem_with_fringe   – download DEM + optional fringe
    2. dem_to_surface_stl      – convert DEM to triangle surface mesh
    3. dem_sheet_to_block      – make a closed 3D solid block
    4. clip_dem_block          – clip the DEM block to the AOI polygon

Usage:
    import dem_tile_export as dte
    arr, _, bb = dte.fetch_dem_with_fringe(...)
    mesh = dte.dem_to_surface_stl(...)
    block = dte.dem_sheet_to_block(...)
    clipped = dte.clip_dem_block(...)
"""

# -- Install if needed: ----------------------------------------
# !pip install earthengine-api geemap trimesh shapely k3d pyproj

import ee, geemap, json, numpy as np, matplotlib.pyplot as plt
import trimesh, k3d
from shapely.geometry import shape, Polygon
from typing import Union, Tuple, Dict
from pathlib import Path
from pyproj import Transformer

# -- GEE login -------------------------------------------------
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()


# ------------------------------------------------------------
# 1 ▸ Fetch DEM as a NumPy array (with optional fringe)
# ------------------------------------------------------------
def fetch_dem_with_fringe(
    geojson: Union[str, Dict],
    dem_id: str = "USGS/SRTMGL1_003",
    scale: float = 30,
    fringe_m: float = 0
) -> Tuple[np.ndarray, ee.Image, Dict]:
    """Download DEM (optionally buffered) as a NumPy array."""
    if isinstance(geojson, str):
        geojson = json.loads(geojson)

    geom = ee.Geometry(geojson)
    if fringe_m != 0:
        geom = geom.buffer(fringe_m)

    img = ee.Image(dem_id).select("elevation").clip(geom)
    region = geom.bounds()
    arr = geemap.ee_to_numpy(img, region=region, scale=scale)
    arr = np.squeeze(arr)

    coords = region.getInfo()["coordinates"][0]
    xs, ys = zip(*coords)
    bbox = dict(xmin=min(xs), xmax=max(xs), ymin=min(ys), ymax=max(ys))
    return arr, img, bbox


# ------------------------------------------------------------
# 2 ▸ DEM array → triangle surface → STL
# ------------------------------------------------------------
def dem_to_surface_stl(
    dem_np: np.ndarray,
    bbox: dict,
    out_stl: str | Path = None,
    vertical_exaggeration: float = 1.0,
    show_k3d: bool = True,
) -> trimesh.Trimesh:
    """Convert DEM array to a triangular surface and save as STL."""
    rows, cols = dem_np.shape
    lon_step = (bbox["xmax"] - bbox["xmin"]) / cols
    lat_step = (bbox["ymax"] - bbox["ymin"]) / rows
    lon = bbox["xmin"] + (np.arange(cols) + 0.5) * lon_step
    lat = bbox["ymax"] - (np.arange(rows) + 0.5) * lat_step
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    z = dem_np * vertical_exaggeration
    mask = ~np.isnan(z)
    verts = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), z.ravel())).astype(np.float32)

    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i00 = r * cols + c
            i10 = i00 + 1
            i01 = i00 + cols
            i11 = i01 + 1
            if mask.ravel()[[i00, i10, i01, i11]].all():
                faces.append([i00, i01, i11])
                faces.append([i00, i11, i10])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)

    if out_stl:
        mesh.export(out_stl)
        print(f"✓ STL written to {Path(out_stl).resolve()}")

    if show_k3d:
        plot = k3d.plot()
        plot += k3d.mesh(
            vertices=mesh.vertices.astype(np.float32),
            indices=mesh.faces.astype(np.uint32),
            wireframe=False,
            color=0xE0E0E0,
            flat_shading=False,
        )
        plot.display()
    return mesh


# ------------------------------------------------------------
# 3 ▸ Surface → watertight block (manifold)
# ------------------------------------------------------------
def dem_sheet_to_block(
    surf_mesh: trimesh.Trimesh,
    rows: int,
    cols: int,
    base_value: float = 50.0,
    base_mode: str = "offset"  # or 'absolute'
) -> trimesh.Trimesh:
    """Convert surface mesh to a watertight solid block."""
    top_v = surf_mesh.vertices
    z_min = top_v[:, 2].min()
    z_base = z_min - abs(base_value) if base_mode == "offset" else float(base_value)

    bot_v = top_v.copy()
    bot_v[:, 2] = z_base
    verts = np.vstack([top_v, bot_v])
    n_top = rows * cols
    faces = surf_mesh.faces.tolist()

    def wall_strip(i0, i1):
        faces.append([i0, i1, i1 + n_top])
        faces.append([i0, i1 + n_top, i0 + n_top])

    for c in range(cols - 1):
        wall_strip(c, c + 1)
    for r in range(rows - 1):
        wall_strip(r * cols + cols - 1, (r + 1) * cols + cols - 1)
    for c in range(cols - 1, 0, -1):
        wall_strip((rows - 1) * cols + c, (rows - 1) * cols + c - 1)
    for r in range(rows - 1, 0, -1):
        wall_strip(r * cols, (r - 1) * cols)

    for r in range(rows - 1):
        for c in range(cols - 1):
            i00 = r * cols + c
            i10 = i00 + 1
            i01 = i00 + cols
            i11 = i01 + 1
            faces.append([i00 + n_top, i11 + n_top, i10 + n_top])
            faces.append([i00 + n_top, i01 + n_top, i11 + n_top])

    block = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    block.remove_duplicate_faces()
    block.merge_vertices()
    block.fix_normals()
    return block


# ------------------------------------------------------------
# 4 ▸ Project, scale, clip DEM to AOI polygon
# ------------------------------------------------------------
def clip_dem_block(
    dem_block: trimesh.Trimesh,
    aoi_geojson: dict,
    out_stl: str | Path = "dem_clipped.stl",
    prism_stl: str | Path = None,
    z_padding: float = 20.0,
    project_crs: str | None = "auto",
    xy_scale: float = 1.0,
    z_scale: float = 1.0,
    show_k3d: bool = True,
) -> trimesh.Trimesh:
    """Clips a watertight DEM block to a GeoJSON AOI polygon (with CRS projection and scaling)."""
    def auto_utm(lon, lat):
        zone = int((lon + 180) / 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"

    centroid = shape(aoi_geojson).centroid
    crs_out = auto_utm(centroid.x, centroid.y) if project_crs == "auto" else project_crs
    transformer = Transformer.from_crs("EPSG:4326", crs_out, always_xy=True)

    dem_v = dem_block.vertices.copy()
    x_proj, y_proj = transformer.transform(dem_v[:, 0], dem_v[:, 1])
    dem_v[:, 0] = x_proj * xy_scale
    dem_v[:, 1] = y_proj * xy_scale
    dem_v[:, 2] *= z_scale
    dem_block_proj = trimesh.Trimesh(vertices=dem_v, faces=dem_block.faces, process=False)

    z0, z1 = dem_block_proj.bounds[0][2], dem_block_proj.bounds[1][2]
    geom = shape(aoi_geojson)
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    prisms = []
    for g in geoms:
        proj_coords = [transformer.transform(*pt) for pt in g.exterior.coords]
        scaled = [(x * xy_scale, y * xy_scale) for x, y in proj_coords]
        prism = trimesh.creation.extrude_polygon(
            Polygon(scaled),
            height=(z1 - z0) + 2 * z_padding,
            transform=trimesh.transformations.translation_matrix([0, 0, z0 - z_padding]),
        )
        prisms.append(prism)

    aoi_prism = trimesh.util.concatenate(prisms) if len(prisms) > 1 else prisms[0]
    if prism_stl:
        aoi_prism.export(prism_stl)
        print(f"✓ AOI prism STL saved: {Path(prism_stl).resolve()}")

    print("⧗ Boolean intersection …")
    clipped = dem_block_proj.intersection(aoi_prism, check_volume=True)

    if clipped.is_empty:
        raise RuntimeError("Boolean result is empty – check input geometry.")

    clipped.export(out_stl)
    print(f"✓ Clipped STL saved: {Path(out_stl).resolve()}")

    if show_k3d:
        plot = k3d.plot()
        plot += k3d.mesh(
            vertices=clipped.vertices.astype(np.float32),
            indices=clipped.faces.astype(np.uint32),
            color=0xCCCCCC,
            flat_shading=False,
        )
        plot.display()

    return clipped
