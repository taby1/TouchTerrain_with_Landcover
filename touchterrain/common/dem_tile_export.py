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

from typing import List, Literal
import math
# import numpy as np
# import trimesh
# from pathlib import Path


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

    # clipped.export(out_stl)
    # print(f"✓ Clipped STL saved: {Path(out_stl).resolve()}")
    if out_stl:
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

# ------------------------------------------------------------
# 5 ▸ Post-process: contour bands → one STL with multiple bodies
# ------------------------------------------------------------

def add_contour_bands_one_stl(
    solid: trimesh.Trimesh,
    *,
    band_thickness_mm: float,
    band_interval_ft: float,
    first_band_elev_ft: float,
    # How your mesh Z-units relate to real meters after clip_dem_block:
    #   mesh_Z = meters * z_scale  → pass z_unit_per_meter = z_scale
    z_unit_per_meter: float,
    # Needed if your block was made with base_mode="absolute":
    #   dem_sheet_to_block(..., base_mode="absolute", base_value=floor_abs_z)
    # Pass that same floor in meters here, so we can place the bands wrt absolute MSL.
    floor_abs_z_m: float | None = None,
    elevation_mode: Literal["absolute", "relative_to_floor"] = "absolute",
    xy_pad_factor: float = 0.05,
    out_stl: str | Path | None = None,
    return_parts: bool = False,
    # Diagnostics / robustness
    debug: bool = True,
    debug_log_path: str | Path | None = None,
    boolean_strict: bool = True,
    attempt_repair_if_open: bool = True,
) -> trimesh.Trimesh | tuple[trimesh.Trimesh, list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    """
    Build horizontal slabs at specified contour elevations, split the watertight tile
    into bands and between-band fills, then concatenate everything into ONE mesh
    (multiple disconnected shells) and optionally export.

    Elevation math:
      - If elevation_mode == "absolute":
          band_center_abs_m(k) = first_band_elev_ft * 0.3048 + k * (band_interval_ft * 0.3048)
        Bands are placed by absolute elevation (MSL), independent of the tile’s floor.
      - If elevation_mode == "relative_to_floor":
          band_center_abs_m(k) = floor_abs_z_m + first_band_elev_ft * 0.3048 + k * (band_interval_ft * 0.3048)
        Useful if your 'first' and 'interval' were meant as offsets above the block floor.

      In both cases, mesh_Z = band_center_abs_m * z_unit_per_meter.

    Units across pipeline:
      - dem_to_surface_stl: Z in meters * vertical_exaggeration
      - dem_sheet_to_block(base_mode='absolute', base_value=floor_abs_z):
            base plane = floor_abs_z (meters)
      - clip_dem_block(..., z_scale=Z_SCALE):
            final mesh Z-units = meters * Z_SCALE  (so pass z_unit_per_meter=Z_SCALE here)
    """
    def _fmt(v: float) -> str:
        return f"{v:,.9f}"

    def _w(msg: str):
        if debug:
            print(msg)
        if debug_log_path is not None:
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    # reset log
    if debug_log_path is not None:
        Path(debug_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(debug_log_path, "w", encoding="utf-8") as f:
            f.write("Contour band diagnostics\n")

    FT_TO_M = 0.3048
    MM_TO_M = 0.001

    # ---- Convert user parameters into meters and mesh units ----
    interval_m = band_interval_ft * FT_TO_M
    thickness_m = band_thickness_mm * MM_TO_M
    first_m = first_band_elev_ft * FT_TO_M

    # Elevation frame selection
    if elevation_mode == "absolute":
        # first_m is literal absolute elevation
        # floor_abs_z_m is not required but is OK to pass (for logging)
        pass
    elif elevation_mode == "relative_to_floor":
        if floor_abs_z_m is None:
            raise ValueError("relative_to_floor mode requires floor_abs_z_m")
        first_m = floor_abs_z_m + first_m
    else:
        raise ValueError("elevation_mode must be 'absolute' or 'relative_to_floor'")

    # Mesh-unit conversions
    interval_mesh = interval_m * z_unit_per_meter
    thickness_mesh = band_thickness_mm
    # thickness_mesh = thickness_m * z_unit_per_meter
    first_mesh = first_m * z_unit_per_meter
    floor_mesh = (floor_abs_z_m * z_unit_per_meter) if floor_abs_z_m is not None else None

    # Guard: degenerate thickness at tiny z_scales
    thickness_min_mesh = 1e-7  # ~0.1 micron in 'meters' units; safe for float
    if thickness_mesh < thickness_min_mesh:
        _w(f"NOTE: requested band thickness {thickness_m:.6f} m becomes {thickness_mesh:.9f} in mesh units; "
           f"clamping slab thickness to {thickness_min_mesh:.9f} for boolean robustness.")
        # We will still use the true [lo,hi] span for band classification; slab box gets min height.
        slab_box_min_height = thickness_min_mesh
    else:
        slab_box_min_height = None  # use true span

    # ---- Basic geometry facts ----
    (xmin, ymin, zmin), (xmax, ymax, zmax) = solid.bounds
    dx, dy = xmax - xmin, ymax - ymin
    cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    pad_x, pad_y = dx * xy_pad_factor, dy * xy_pad_factor
    slab_x, slab_y = dx + 2 * pad_x, dy + 2 * pad_y
    total_h = zmax - zmin

    # ---- Diagnostics header ----
    _w("── Mesh Z diagnostics (mesh units = meters × z_scale) ──")
    _w(f"Z_min(mesh): {_fmt(zmin)}   Z_max(mesh): {_fmt(zmax)}   Height(mesh): {_fmt(total_h)}")
    if floor_mesh is not None:
        _w(f"Floor_abs(mesh): {_fmt(floor_mesh)}  (meters: {floor_abs_z_m:.3f})")
    _w(f"Interval: {interval_m:.3f} m ({band_interval_ft} ft)  → mesh {_fmt(interval_mesh)}")
    _w(f"Band thickness: {thickness_m:.3f} m ({band_thickness_mm} mm)  → mesh {_fmt(thickness_mesh)}")
    _w(f"First band center: {first_m:.3f} m ({first_band_elev_ft} ft)  → mesh {_fmt(first_mesh)}")
    _w(f"Elevation mode: {elevation_mode}")

    # ---- Compute band centers that could intersect [zmin, zmax] ----
    # A band centered at c has span [c - t/2, c + t/2] in mesh units.
    half_t_mesh = 0.5 * thickness_mesh

    # Determine k range so that band span intersects the tile’s Z range.
    # Require: (c + t/2) >= zmin  and  (c - t/2) <= zmax
    #   c = first_mesh + k * interval_mesh
    if interval_mesh <= 0:
        raise ValueError("band_interval_ft must be positive after scaling.")
    k_start = math.ceil((zmin - (first_mesh + half_t_mesh)) / interval_mesh)
    k_end   = math.floor((zmax - (first_mesh - half_t_mesh)) / interval_mesh)

    centers_mesh: List[float] = [first_mesh + k * interval_mesh for k in range(k_start, k_end + 1)]
    # Band spans (mesh)
    band_spans_mesh: List[Tuple[float, float]] = [
        (max(c - half_t_mesh, zmin), min(c + half_t_mesh, zmax)) for c in centers_mesh
    ]
    # Filter any degenerate after clipping
    band_spans_mesh = [(lo, hi) for (lo, hi) in band_spans_mesh if hi > lo]

    # Print the centers (mesh + meters + feet)
    _w("── Band centers ──")
    if not centers_mesh:
        _w("No bands intersect this tile’s Z extent.")
    else:
        for i, c_mesh in enumerate(centers_mesh[:8]):
            c_m = c_mesh / z_unit_per_meter
            c_ft = c_m / FT_TO_M
            _w(f"#{i:02d} center: mesh {_fmt(c_mesh)}  |  {c_m:.3f} m  |  {c_ft:.2f} ft")
        if len(centers_mesh) > 8:
            _w(" ...")
            for i, c_mesh in enumerate(centers_mesh[-3:], start=len(centers_mesh)-3):
                c_m = c_mesh / z_unit_per_meter
                c_ft = c_m / FT_TO_M
                _w(f"#{i:02d} center: mesh {_fmt(c_mesh)}  |  {c_m:.3f} m  |  {c_ft:.2f} ft")

    # Build full coverage of [zmin, zmax]: fill, band, fill, ...
    intervals: List[Tuple[str, Tuple[float, float]]] = []
    cursor = zmin
    for (blo, bhi) in band_spans_mesh:
        if blo > cursor:
            intervals.append(("fill", (cursor, blo)))
        intervals.append(("band", (blo, bhi)))
        cursor = bhi
    if cursor < zmax:
        intervals.append(("fill", (cursor, zmax)))

    # Slab list printout
    _w("── Slab Z-intervals (mesh | meters | feet) ──")
    _w(f"Total intervals: {len(intervals)} "
       f"(bands={sum(1 for k,_ in intervals if k=='band')}, fills={sum(1 for k,_ in intervals if k=='fill')})")
    show_n = min(10, len(intervals))
    for idx, (kind, (z0, z1)) in enumerate(intervals[:show_n]):
        dz = z1 - z0
        z0_m, z1_m, dz_m = z0 / z_unit_per_meter, z1 / z_unit_per_meter, dz / z_unit_per_meter
        z0_ft, z1_ft, dz_ft = z0_m / FT_TO_M, z1_m / FT_TO_M, dz_m / FT_TO_M
        _w(f"#{idx:02d} {kind:5s}: mesh [{_fmt(z0)} .. {_fmt(z1)}] Δ={_fmt(dz)}"
           f"  |  m [{z0_m:.3f} .. {z1_m:.3f}] Δ={dz_m:.3f}"
           f"  |  ft [{z0_ft:.2f} .. {z1_ft:.2f}] Δ={dz_ft:.2f}")
    if len(intervals) > show_n:
        _w(" ...")

    # Operand diagnostics and optional repair
    def _stats(m: trimesh.Trimesh, label: str):
        try:
            vol = float(m.volume)
        except Exception:
            vol = float('nan')
        _w(f"[{label}] watertight={m.is_watertight}, bodies={m.body_count}, "
           f"faces={len(m.faces):,}, verts={len(m.vertices):,}, euler={m.euler_number}, "
           f"volume={vol if np.isfinite(vol) else 'nan'}")

    _w("── Operand diagnostics ──")
    _stats(solid, "solid (pre)")
    if attempt_repair_if_open and not solid.is_watertight:
        _w("Attempting light repair (duplicate faces, unreferenced verts, fill_holes, merge, fix_normals) …")
        try:
            solid.remove_duplicate_faces()
            solid.remove_unreferenced_vertices()
            trimesh.repair.fill_holes(solid)
            solid.merge_vertices()
            solid.fix_normals()
        except Exception as e:
            _w(f"Repair raised: {e!r}")
        _stats(solid, "solid (post-repair)")

    def make_slab(z0: float, z1: float) -> trimesh.Trimesh:
        thickness = z1 - z0
        if slab_box_min_height is not None and thickness < slab_box_min_height:
            center = 0.5 * (z0 + z1)
            thickness = slab_box_min_height
            z0 = center - 0.5 * thickness
            z1 = center + 0.5 * thickness
        center_z = 0.5 * (z0 + z1)
        return trimesh.creation.box(
            extents=[slab_x, slab_y, max(1e-9, thickness)],
            transform=trimesh.transformations.translation_matrix([cx, cy, center_z]),
        )

    band_parts: List[trimesh.Trimesh] = []
    fill_parts: List[trimesh.Trimesh] = []

    for idx, (kind, (z0, z1)) in enumerate(intervals):
        slab = make_slab(z0 - (1 if idx == 0 else 0), z1 + (1 if idx == len(intervals) - 1 else 0))
        _stats(slab, f"slab#{idx}:{kind}")
        try:
            piece = solid.intersection(slab, check_volume=boolean_strict)
        except ValueError as e:
            _w(f"Boolean intersection failed at slab#{idx} ({kind}) → {e}")
            if boolean_strict:
                _w("Hint: set boolean_strict=False to allow non-volume operands; "
                   "or inspect watertightness above.")
            raise

        if piece.is_empty:
            _w(f"Note: slab#{idx} ({kind}) produced empty piece; skipping.")
            continue

        piece.remove_duplicate_faces()
        piece.merge_vertices()
        piece.fix_normals()

        (band_parts if kind == "band" else fill_parts).append(piece)

    merged = trimesh.util.concatenate([*band_parts, *fill_parts])
    _stats(merged, "merged (final)")

    if out_stl:
        out_stl = Path(out_stl)
        merged.export(out_stl)
        _w(f"✓ Single STL with multi-bodies (bands+fills): {out_stl.resolve()}")

    return (merged, band_parts, fill_parts) if return_parts else merged
