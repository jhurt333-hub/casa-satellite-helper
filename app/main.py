from __future__ import annotations

import asyncio
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from netCDF4 import Dataset, num2date
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class Settings:
    bucket: str = os.getenv("GOES_BUCKET", "noaa-goes19")
    product: str = os.getenv("GOES_PRODUCT", "ABI-L2-ACHTF")
    center_lat: float = float(os.getenv("CASA_LAT", "17.97"))
    center_lon: float = float(os.getenv("CASA_LON", "-87.93"))
    default_north: float = float(os.getenv("BBOX_NORTH", "18.35"))
    default_south: float = float(os.getenv("BBOX_SOUTH", "17.45"))
    default_west: float = float(os.getenv("BBOX_WEST", "-88.15"))
    default_east: float = float(os.getenv("BBOX_EAST", "-86.90"))
    cold_k: float = float(os.getenv("COLD_THRESHOLD_K", "235"))
    deep_k: float = float(os.getenv("DEEP_THRESHOLD_K", "215"))
    max_points: int = int(os.getenv("MAX_POINTS", "400"))
    cache_seconds: int = int(os.getenv("CACHE_SECONDS", "240"))
    api_key: str = os.getenv("API_KEY", "")


settings = Settings()
app = FastAPI(title="Casa GOES-19 Satellite Helper", version="1.0.0")
s3 = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED, retries={"max_attempts": 3}))
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_lock = asyncio.Lock()


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def parse_scan_time(key: str) -> datetime | None:
    # NOAA filename segment: _sYYYYJJJHHMMSSd (UTC start time).
    try:
        token = Path(key).name.split("_s", 1)[1].split("_", 1)[0]
        return datetime.strptime(token[:13], "%Y%j%H%M%S").replace(tzinfo=timezone.utc)
    except (IndexError, ValueError):
        return None


def latest_object(now: datetime | None = None) -> tuple[str, datetime | None]:
    now = now or datetime.now(timezone.utc)
    for hours_back in range(8):
        hour = now - timedelta(hours=hours_back)
        prefix = f"{settings.product}/{hour:%Y}/{hour:%j}/{hour:%H}/"
        page = s3.list_objects_v2(Bucket=settings.bucket, Prefix=prefix)
        objects = [o for o in page.get("Contents", []) if o["Key"].endswith(".nc")]
        if objects:
            newest = max(objects, key=lambda item: item.get("LastModified", datetime.min.replace(tzinfo=timezone.utc)))
            key = newest["Key"]
            return key, parse_scan_time(key)
    raise RuntimeError("No recent ACHTF NetCDF file found in the last 8 UTC hours")


def projection(dataset: Dataset) -> tuple[Transformer, Transformer, float]:
    p = dataset.variables["goes_imager_projection"]
    height = float(p.perspective_point_height)
    crs = CRS.from_proj4(
        f"+proj=geos +h={height} +lon_0={float(p.longitude_of_projection_origin)} "
        f"+sweep={p.sweep_angle_axis} +a={float(p.semi_major_axis)} "
        f"+b={float(p.semi_minor_axis)} +units=m +no_defs"
    )
    wgs84 = CRS.from_epsg(4326)
    return (
        Transformer.from_crs(wgs84, crs, always_xy=True),
        Transformer.from_crs(crs, wgs84, always_xy=True),
        height,
    )


def nearest_slice(axis: np.ndarray, low: float, high: float, pad: int = 2) -> slice:
    lo, hi = sorted((low, high))
    indices = np.flatnonzero((axis >= lo) & (axis <= hi))
    if not indices.size:
        raise ValueError("Requested box is outside the ABI fixed grid")
    return slice(max(0, int(indices[0]) - pad), min(axis.size, int(indices[-1]) + pad + 1))


def haversine_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    p1, p2 = np.radians(lat), math.radians(lat0)
    dp = p1 - p2
    dl = np.radians(lon - lon0)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def cluster_storm_cells(
    cold: np.ndarray,
    temp: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    deep_k: float,
    center_lat: float,
    center_lon: float,
    min_pixels: int = 3,
) -> list[dict]:
    visited = np.zeros(cold.shape, dtype=bool)
    storm_cells = []

    neighbor_offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    height, width = cold.shape

    for start_row, start_col in zip(*np.where(cold)):
        if visited[start_row, start_col]:
            continue

        stack = [(int(start_row), int(start_col))]
        visited[start_row, start_col] = True
        component = []

        while stack:
            row, col = stack.pop()
            component.append((row, col))

            for row_offset, col_offset in neighbor_offsets:
                next_row = row + row_offset
                next_col = col + col_offset

                if (
                    next_row < 0
                    or next_row >= height
                    or next_col < 0
                    or next_col >= width
                    or visited[next_row, next_col]
                    or not cold[next_row, next_col]
                ):
                    continue

                visited[next_row, next_col] = True
                stack.append((next_row, next_col))

        if len(component) < min_pixels:
            continue
 
        component_rows = np.asarray(
            [item[0] for item in component],
            dtype=int,
        )
        component_cols = np.asarray(
            [item[1] for item in component],
            dtype=int,
        )

        component_lat = lat[
            component_rows,
            component_cols
        ]
        component_lon = lon[
            component_rows,
            component_cols
        ]
        component_temp = temp[
            component_rows,
            component_cols
        ]

        finite = (
            np.isfinite(component_lat)
            & np.isfinite(component_lon)
            & np.isfinite(component_temp)
        )

        if not finite.any():
            continue

        component_lat = component_lat[finite]
        component_lon = component_lon[finite]
        component_temp = component_temp[finite]

        center_cell_lat = float(
            np.mean(component_lat)
        )
        center_cell_lon = float(
            np.mean(component_lon)
        )

        distance_km = float(
            haversine_km(
                np.asarray([center_cell_lat]),
                np.asarray([center_cell_lon]),
                center_lat,
                center_lon,
            )[0]
        )

        latitude_difference = math.radians(
            center_cell_lat - center_lat
        )
        longitude_difference = math.radians(
            center_cell_lon - center_lon
        )

        bearing_y = (
            math.sin(longitude_difference)
            * math.cos(math.radians(center_cell_lat))
        )

        bearing_x = (
            math.cos(math.radians(center_lat))
            * math.sin(math.radians(center_cell_lat))
            - math.sin(math.radians(center_lat))
            * math.cos(math.radians(center_cell_lat))
            * math.cos(longitude_difference)
        )

        bearing_degrees = (
            math.degrees(
                math.atan2(bearing_y, bearing_x)
            )
            + 360
        ) % 360

        coldest_k = float(
            np.min(component_temp)
        )
        deep_pixels = int(
            np.sum(component_temp < deep_k)
        )
        pixel_count = int(component_temp.size)

        storm_cells.append({
            "cell_id":
                f"cell-{len(storm_cells) + 1}",
            "center_lat":
                round(center_cell_lat, 4),
            "center_lon":
                round(center_cell_lon, 4),
            "pixel_count": pixel_count,
            "deep_pixel_count": deep_pixels,
            "approx_area_km2":
                round(pixel_count * 4.0, 1),
            "coldest_k":
                round(coldest_k, 1),
            "coldest_c":
                round(coldest_k - 273.15, 1),
            "mean_cloud_top_k":
                round(float(np.mean(component_temp)), 1),
            "distance_from_casa_km":
                round(distance_km, 1),
            "distance_from_casa_miles":
                round(distance_km * 0.621371, 1),
            "bearing_from_casa_deg":
                round(bearing_degrees, 1),
            "bbox": {
                "south":
                    round(float(np.min(component_lat)), 4),
                "west":
                    round(float(np.min(component_lon)), 4),
                "north":
                    round(float(np.max(component_lat)), 4),
                "east":
                    round(float(np.max(component_lon)), 4),
            },
        })

    storm_cells.sort(
        key=lambda cell: (
            -cell["deep_pixel_count"],
            cell["coldest_k"],
            -cell["pixel_count"],
        )
    )

    return storm_cells

def observed_at(dataset: Dataset, fallback: datetime | None) -> str | None:
    if "t" in dataset.variables:
        var = dataset.variables["t"]
        dt = num2date(var[:].item(), var.units, only_use_cftime_datetimes=False)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return fallback.isoformat().replace("+00:00", "Z") if fallback else None


def parse_file(path: str, key: str, scan_time: datetime | None, box: tuple[float, float, float, float], cold_k: float, deep_k: float, max_points: int) -> dict[str, Any]:
    south, west, north, east = box
    with Dataset(path, "r") as ds:
        to_grid, to_geo, height = projection(ds)
        corner_lon = np.array([west, east, west, east])
        corner_lat = np.array([south, south, north, north])
        gx, gy = to_grid.transform(corner_lon, corner_lat)
        xs = np.asarray(ds.variables["x"][:])
        ys = np.asarray(ds.variables["y"][:])
        x_slice = nearest_slice(xs, float(np.nanmin(gx) / height), float(np.nanmax(gx) / height))
        y_slice = nearest_slice(ys, float(np.nanmin(gy) / height), float(np.nanmax(gy) / height))

        x_grid, y_grid = np.meshgrid(xs[x_slice] * height, ys[y_slice] * height)
        lon, lat = to_geo.transform(x_grid, y_grid)
        temp = np.ma.asarray(ds.variables["TEMP"][y_slice, x_slice]).filled(np.nan).astype(float)
        valid = np.isfinite(temp) & np.isfinite(lat) & np.isfinite(lon)
        valid &= (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
        cold = valid & (temp <= cold_k)
        rows, cols = np.where(cold)

        # Preserve the coldest pixels if the response limit is reached.
        if rows.size > max_points:
            chosen = np.argpartition(temp[rows, cols], max_points - 1)[:max_points]
            rows, cols = rows[chosen], cols[chosen]
        order = np.argsort(temp[rows, cols]) if rows.size else np.array([], dtype=int)
        rows, cols = rows[order], cols[order]
        distances = haversine_km(lat[rows, cols], lon[rows, cols], settings.center_lat, settings.center_lon)

        points = []
        for row, col, distance in zip(rows, cols, distances):
            k = float(temp[row, col])
            points.append({
                "row": int(row),
                "col": int(col),
                "lat": round(float(lat[row, col]), 4),
                "lon": round(float(lon[row, col]), 4),
                "cloud_top_k": round(k, 1),
                "cloud_top_c": round(k - 273.15, 1),
                "class": "deep" if k <= deep_k else "cold",
                "distance_from_casa_km": round(float(distance), 1),
                "distance_from_casa_miles": round(float(distance) * 0.621371, 1),
            })

        storm_cells = cluster_storm_cells(
                    cold=cold,
                    temp=temp,
                    lat=lat,
                    lon=lon,
                    deep_k=deep_k,
                    center_lat=settings.center_lat,
                    center_lon=settings.center_lon,
                )        
        
        valid_temps = temp[valid]
        return {
                    "observed_at": observed_at(ds, scan_time),
                    "source": {"bucket": settings.bucket, "key": key, "satellite": "GOES-19", "product": settings.product},
                    "center": {"name": "Casa de Rasta / Secret Beach", "lat": settings.center_lat, "lon": settings.center_lon},
                    "bbox": {"south": south, "west": west, "north": north, "east": east},
                    "thresholds_k": {"cold": cold_k, "deep": deep_k},
                    "summary": {
                        "valid_pixels": int(valid.sum()),
                        "cold_pixels": int(cold.sum()),
                        "deep_pixels": int((valid & (temp <= deep_k)).sum()),
                        "coldest_k": round(float(np.nanmin(valid_temps)), 1) if valid_temps.size else None,
                        "coldest_c": round(float(np.nanmin(valid_temps)) - 273.15, 1) if valid_temps.size else None,
                        "points_returned": len(points),
                        "points_truncated": int(cold.sum()) > len(points),
                    },
                    "storm_cells": storm_cells,
                    "cells": points,
                }


def fetch_and_parse(box: tuple[float, float, float, float], cold_k: float, deep_k: float, max_points: int) -> dict[str, Any]:
    key, scan_time = latest_object()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "latest.nc")
        s3.download_file(settings.bucket, key, path)
        return parse_file(path, key, scan_time, box, cold_k, deep_k, max_points)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/convection", dependencies=[Depends(require_key)])
async def convection(
    south: float = Query(default=settings.default_south, ge=-90, le=90),
    west: float = Query(default=settings.default_west, ge=-180, le=180),
    north: float = Query(default=settings.default_north, ge=-90, le=90),
    east: float = Query(default=settings.default_east, ge=-180, le=180),
    cold_k: float = Query(default=settings.cold_k, ge=180, le=300),
    deep_k: float = Query(default=settings.deep_k, ge=180, le=300),
    max_points: int = Query(default=settings.max_points, ge=1, le=2000),
) -> JSONResponse:
    if south >= north or west >= east or deep_k > cold_k:
        raise HTTPException(status_code=422, detail="Require south < north, west < east, and deep_k <= cold_k")
    if (north - south) > 5 or (east - west) > 5:
        raise HTTPException(status_code=422, detail="Bounding box may span at most 5 degrees per axis")
    cache_key = repr((south, west, north, east, cold_k, deep_k, max_points))
    cached = _cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < settings.cache_seconds:
        return JSONResponse(cached[1], headers={"Cache-Control": f"public, max-age={settings.cache_seconds}", "X-Cache": "HIT"})
    async with _lock:
        cached = _cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < settings.cache_seconds:
            return JSONResponse(cached[1], headers={"X-Cache": "HIT"})
        try:
            result = await asyncio.to_thread(fetch_and_parse, (south, west, north, east), cold_k, deep_k, max_points)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Satellite processing failed: {exc}") from exc
        _cache[cache_key] = (time.monotonic(), result)
        return JSONResponse(result, headers={"Cache-Control": f"public, max-age={settings.cache_seconds}", "X-Cache": "MISS"})
