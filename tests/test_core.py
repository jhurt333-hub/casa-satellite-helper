from datetime import timezone

import numpy as np

from app.main import haversine_km, nearest_slice, parse_scan_time


def test_scan_time_from_noaa_filename():
    dt = parse_scan_time("ABI-L2-ACHTF/2026/225/01/OR_ABI-L2-ACHTF-M6_G19_s20262250100211_e.nc")
    assert dt.tzinfo == timezone.utc
    assert dt.strftime("%Y-%j %H:%M:%S") == "2026-225 01:00:21"


def test_nearest_slice_handles_descending_axis():
    axis = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    result = nearest_slice(axis, 1.5, 3.5, pad=0)
    assert np.array_equal(axis[result], np.array([3.0, 2.0]))


def test_haversine_zero_at_casa():
    result = haversine_km(np.array([17.97]), np.array([-87.93]), 17.97, -87.93)
    assert result[0] == 0
