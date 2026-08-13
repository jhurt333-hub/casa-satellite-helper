# Casa GOES-19 satellite helper

A small HTTP service that finds NOAA's latest GOES-19 ABI L2 full-disk Cloud Top Temperature (`ACHTF`) file, reads only the requested Belize-area grid window, geolocates its ABI fixed-grid pixels, and returns compact JSON for convection tracking.

The default box is intentionally asymmetric around Casa de Rasta (`17.97,-87.93`), extending farther east and south over the Caribbean. `235 K` (~`-38.2 C`) marks cold cloud tops and `215 K` (~`-58.2 C`) marks deep convection. These are configurable screening thresholds, not a storm diagnosis.

## Run

```bash
cp .env.example .env
docker build -t casa-satellite-helper .
docker run --rm --env-file .env -p 8080:8080 casa-satellite-helper
```

```bash
curl -H "X-API-Key: replace-with-a-long-random-value" \
  "http://localhost:8080/v1/convection"
```

Interactive API documentation is at `/docs`; liveness is at `/health` and does not require a key.

Optional query parameters: `south`, `west`, `north`, `east`, `cold_k`, `deep_k`, and `max_points`. Boxes are limited to 5 degrees per axis and responses to 2,000 points. The coldest points are retained when the result is truncated.

## Deploy

Deploy the included container to any platform that accepts a Dockerfile (Cloud Run, Fly.io, Railway, Render, ECS, etc.). Give it at least **512 MB RAM** and about **100 MB writable temporary disk**. The NOAA bucket is public; no AWS credentials are needed. Set `API_KEY` in the platform's secret manager and configure all other values from `.env.example` as ordinary environment variables.

The service downloads one roughly 25–35 MB NetCDF file on a cache miss. A single-process deployment is recommended because the cache and request lock are in-process. For stronger production caching across replicas, put the Worker cache in front or add Redis/object storage.

## Response shape

```json
{
  "observed_at": "2026-08-13T01:04:36Z",
  "source": {"satellite": "GOES-19", "product": "ABI-L2-ACHTF", "key": "..."},
  "center": {"name": "Casa de Rasta / Secret Beach", "lat": 17.97, "lon": -87.93},
  "bbox": {"south": 17.45, "west": -88.15, "north": 18.35, "east": -86.9},
  "thresholds_k": {"cold": 235, "deep": 215},
  "summary": {"valid_pixels": 0, "cold_pixels": 0, "deep_pixels": 0, "coldest_k": null},
  "cells": [
    {"lat": 17.8, "lon": -87.2, "cloud_top_k": 208.4, "cloud_top_c": -64.8, "class": "deep", "distance_from_casa_km": 79.1, "distance_from_casa_miles": 49.1}
  ]
}
```

Copy `worker-example.js` into the Casa Worker and store its key with `wrangler secret put SATELLITE_HELPER_KEY`. Do not put the key directly in Worker source.

## Operational notes

- The helper checks the latest eight UTC hour prefixes because NOAA keys are organized as `product/YYYY/JJJ/HH/`.
- The source timestamp comes from the NetCDF `t` coordinate, with the filename scan start as fallback.
- `cells` are individual native ~2 km product pixels, sorted coldest first. Pixel counts are useful for change detection; tracking direction and ETA requires retaining several observations in the Casa Worker or a database.
- Treat output as situational awareness only. It is not an official warning product; Casa should continue to surface Belize/NHC/NMS alerts independently.
