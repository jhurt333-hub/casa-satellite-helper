from __future__ import annotations

import asyncio
import math
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from netCDF4 import Dataset, num2date
from pyproj import CRS, Transformer


BUCKET = os.getenv("GOES_BUCKET", "noaa-goes19")
PRODUCT = os.getenv("GOES_RAIN_PRODUCT", "ABI-L2-RRQPEF")
CASA_LAT = float(os.getenv("CASA_LAT", "17.975"))
CASA_LON = float(os.getenv("CASA_LON", "-87.958056"))
CACHE_SECONDS = int(os.getenv("RAIN_CACHE_SECONDS", "240"))
API_KEY = os.getenv("API_KEY", "")

router = APIRouter()
s3 = boto3.client(
    "s3",
    config=BotoConfig(signature_version=UNSIGNED, retries={"max_attempts": 3}),
)
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_lock = asyncio.Lock()


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def parse_scan_time(key: str) -> datetime | None:
    try:
        token = Path(key).name.split("_s", 1)[1].split("_", 1)[0]
        return datetime.strptime(token[:13], "%Y%j%H%M%S").replace(tzinfo=timezone.utc)
    except (IndexError, ValueError):
        return None


def latest_rain_object(now: datetime | None = None) -> tuple[str, datetime | None]:
    now = now or datetime.now(timezone.utc)
    for hours_back in range(8):
        hour = now - timedelta(hours=hours_back)
        prefix = f"{PRODUCT}/{hour:%Y}/{hour:%j}/{hour:%H}/"
        page = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        objects = [item for item in page.get("Contents", []) if item["Key"].endswith(".nc")]
        if objects:
            newest = max(
                objects,
                key=lambda item: item.get(
                    "LastModified", datetime.min.replace(tzinfo=timezone.utc)
                ),
            )
            return newest["Key"], parse_scan_time(newest["Key"])
    raise RuntimeError("No recent GOES-19 rainfall-rate file found in the last 8 UTC hours")


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


def observed_at(dataset: Dataset, fallback: datetime | None) -> str | None:
    if "t" in dataset.variables:
        var = dataset.variables["t"]
        dt = num2date(var[:].item(), var.units, only_use_cftime_datetimes=False)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return fallback.isoformat().replace("+00:00", "Z") if fallback else None


def rain_band(rate: float) -> str:
    # Service display bands. Raw mm/hour is always returned for transparency.
    if rate >= 25:
        return "intense_estimate"
    if rate >= 5:
        return "heavy_estimate"
    if rate >= 1:
        return "rain_detected"
    if rate >= 0.1:
        return "trace_estimate"
    return "none_detected"


def cluster_rain_areas(
    raining: np.ndarray,
    rate: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> list[dict[str, Any]]:
    visited = np.zeros(raining.shape, dtype=bool)
    areas: list[dict[str, Any]] = []
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    height, width = raining.shape

    for start_row, start_col in zip(*np.where(raining)):
        if visited[start_row, start_col]:
            continue
        stack = [(int(start_row), int(start_col))]
        visited[start_row, start_col] = True
        component: list[tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            component.append((row, col))
            for dr, dc in neighbors:
                nr, nc = row + dr, col + dc
                if (
                    nr < 0 or nr >= height or nc < 0 or nc >= width
                    or visited[nr, nc] or not raining[nr, nc]
                ):
                    continue
                visited[nr, nc] = True
                stack.append((nr, nc))

        rows = np.asarray([item[0] for item in component], dtype=int)
        cols = np.asarray([item[1] for item in component], dtype=int)
        rates = rate[rows, cols]
        lats = lat[rows, cols]
        lons = lon[rows, cols]
        finite = np.isfinite(rates) & np.isfinite(lats) & np.isfinite(lons)
        if not finite.any():
            continue
        rates, lats, lons = rates[finite], lats[finite], lons[finite]
        center_lat, center_lon = float(np.mean(lats)), float(np.mean(lons))
        distance_km = float(
            haversine_km(
                np.asarray([center_lat]), np.asarray([center_lon]), CASA_LAT, CASA_LON
            )[0]
        )
        maximum = float(np.max(rates))
        areas.append({
            "area_id": f"rain-{len(areas) + 1}",
            "center_lat": round(center_lat, 4),
            "center_lon": round(center_lon, 4),
            "pixel_count": int(rates.size),
            "approx_area_km2": round(float(rates.size) * 4.0, 1),
            "max_rate_mm_hr": round(maximum, 2),
            "mean_rate_mm_hr": round(float(np.mean(rates)), 2),
            "band": rain_band(maximum),
            "distance_from_casa_miles": round(distance_km * 0.621371, 1),
            "bbox": {
                "south": round(float(np.min(lats)), 4),
                "west": round(float(np.min(lons)), 4),
                "north": round(float(np.max(lats)), 4),
                "east": round(float(np.max(lons)), 4),
            },
        })

    areas.sort(key=lambda area: (-area["max_rate_mm_hr"], area["distance_from_casa_miles"]))
    return areas


def parse_rain_file(
    path: str,
    key: str,
    scan_time: datetime | None,
    box: tuple[float, float, float, float],
    rain_threshold: float,
    max_points: int,
    include_degraded: bool,
) -> dict[str, Any]:
    south, west, north, east = box
    with Dataset(path, "r") as ds:
        to_grid, to_geo, height = projection(ds)
        corner_lon = np.asarray([west, east, west, east])
        corner_lat = np.asarray([south, south, north, north])
        gx, gy = to_grid.transform(corner_lon, corner_lat)
        xs = np.asarray(ds.variables["x"][:])
        ys = np.asarray(ds.variables["y"][:])
        x_slice = nearest_slice(xs, float(np.nanmin(gx) / height), float(np.nanmax(gx) / height))
        y_slice = nearest_slice(ys, float(np.nanmin(gy) / height), float(np.nanmax(gy) / height))

        x_grid, y_grid = np.meshgrid(xs[x_slice] * height, ys[y_slice] * height)
        lon, lat = to_geo.transform(x_grid, y_grid)
        rate = np.ma.asarray(ds.variables["RRQPE"][y_slice, x_slice]).filled(np.nan).astype(float)
        dqf = np.ma.asarray(ds.variables["DQF"][y_slice, x_slice]).filled(255).astype(int)
        allowed_quality = (dqf == 0) | (include_degraded & (dqf == 2))
        valid = np.isfinite(rate) & np.isfinite(lat) & np.isfinite(lon) & allowed_quality
        valid &= (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
        raining = valid & (rate >= rain_threshold)

        distances = haversine_km(lat, lon, CASA_LAT, CASA_LON)
        casa_candidates = np.where(valid, distances, np.inf)
        if np.isfinite(casa_candidates).any():
            casa_row, casa_col = np.unravel_index(np.argmin(casa_candidates), casa_candidates.shape)
            casa_rate = float(rate[casa_row, casa_col])
            nearest_casa = {
                "lat": round(float(lat[casa_row, casa_col]), 4),
                "lon": round(float(lon[casa_row, casa_col]), 4),
                "distance_miles": round(float(distances[casa_row, casa_col]) * 0.621371, 1),
                "rate_mm_hr": round(casa_rate, 2),
                "rain_detected": casa_rate >= rain_threshold,
                "band": rain_band(casa_rate),
                "quality": "good" if dqf[casa_row, casa_col] == 0 else "degraded",
            }
        else:
            nearest_casa = None

        rows, cols = np.where(raining)
        if rows.size > max_points:
            chosen = np.argpartition(rate[rows, cols], -max_points)[-max_points:]
            rows, cols = rows[chosen], cols[chosen]
        order = np.argsort(rate[rows, cols])[::-1] if rows.size else np.asarray([], dtype=int)
        rows, cols = rows[order], cols[order]
        points = [{
            "lat": round(float(lat[row, col]), 4),
            "lon": round(float(lon[row, col]), 4),
            "rate_mm_hr": round(float(rate[row, col]), 2),
            "band": rain_band(float(rate[row, col])),
            "quality": "good" if dqf[row, col] == 0 else "degraded",
            "distance_from_casa_miles": round(float(distances[row, col]) * 0.621371, 1),
        } for row, col in zip(rows, cols)]

        valid_rates = rate[valid]
        raining_rates = rate[raining]
        return {
            "observed_at": observed_at(ds, scan_time),
            "source": {
                "bucket": BUCKET,
                "key": key,
                "satellite": "GOES-19",
                "product": PRODUCT,
                "variable": "RRQPE",
                "units": "mm/hour",
            },
            "center": {"name": "Casa de Rasta / Secret Beach", "lat": CASA_LAT, "lon": CASA_LON},
            "bbox": {"south": south, "west": west, "north": north, "east": east},
            "threshold_mm_hr": rain_threshold,
            "quality": {"accepted": [0, 2] if include_degraded else [0], "include_degraded": include_degraded},
            "summary": {
                "valid_pixels": int(valid.sum()),
                "rain_pixels": int(raining.sum()),
                "rain_detected_in_box": bool(raining.any()),
                "max_rate_mm_hr": round(float(np.max(raining_rates)), 2) if raining_rates.size else 0.0,
                "mean_raining_rate_mm_hr": round(float(np.mean(raining_rates)), 2) if raining_rates.size else 0.0,
                "mean_all_valid_rate_mm_hr": round(float(np.mean(valid_rates)), 2) if valid_rates.size else None,
                "points_returned": len(points),
                "points_truncated": int(raining.sum()) > len(points),
            },
            "nearest_casa": nearest_casa,
            "rain_areas": cluster_rain_areas(raining, rate, lat, lon),
            "points": points,
            "interpretation": "Satellite-estimated instantaneous rain rate; confirm impacts with local gauges and official warnings.",
        }


def fetch_and_parse_rain(
    box: tuple[float, float, float, float],
    rain_threshold: float,
    max_points: int,
    include_degraded: bool,
) -> dict[str, Any]:
    key, scan_time = latest_rain_object()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "rain.nc")
        s3.download_file(BUCKET, key, path)
        return parse_rain_file(path, key, scan_time, box, rain_threshold, max_points, include_degraded)


@router.get("/v1/rainfall", dependencies=[])
async def rainfall(
    x_api_key: str | None = Header(default=None),
    south: float = Query(default=17.45, ge=-90, le=90),
    west: float = Query(default=-88.25, ge=-180, le=180),
    north: float = Query(default=18.35, ge=-90, le=90),
    east: float = Query(default=-86.90, ge=-180, le=180),
    rain_threshold: float = Query(default=1.0, ge=0.0, le=100.0),
    max_points: int = Query(default=300, ge=1, le=2000),
    include_degraded: bool = Query(default=False),
) -> JSONResponse:
    require_key(x_api_key)
    if south >= north or west >= east:
        raise HTTPException(status_code=422, detail="Require south < north and west < east")
    if (north - south) > 5 or (east - west) > 5:
        raise HTTPException(status_code=422, detail="Bounding box may span at most 5 degrees per axis")
    cache_key = repr((south, west, north, east, rain_threshold, max_points, include_degraded))
    cached = _cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return JSONResponse(cached[1], headers={"Cache-Control": f"public, max-age={CACHE_SECONDS}", "X-Cache": "HIT"})
    async with _lock:
        cached = _cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
            return JSONResponse(cached[1], headers={"X-Cache": "HIT"})
        try:
            result = await asyncio.to_thread(
                fetch_and_parse_rain,
                (south, west, north, east),
                rain_threshold,
                max_points,
                include_degraded,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Rainfall processing failed: {exc}") from exc
        _cache[cache_key] = (time.monotonic(), result)
        return JSONResponse(result, headers={"Cache-Control": f"public, max-age={CACHE_SECONDS}", "X-Cache": "MISS"})
