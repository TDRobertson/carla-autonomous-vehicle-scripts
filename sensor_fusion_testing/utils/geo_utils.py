"""
GNSS to Local ENU (East-North-Up) Coordinate Conversion Utilities

Converts GNSS latitude/longitude/altitude to local meters using the first
sample as the reference origin. Uses the equirectangular approximation which
is accurate for small distances (< 100 km from reference).

This allows the victim to work purely with their GPS sensor output without
needing any ground-truth position reference.
"""

import math
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# WGS84 ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_B = 6356752.314245  # Semi-minor axis (meters)
WGS84_E2 = 1 - (WGS84_B ** 2) / (WGS84_A ** 2)  # Eccentricity squared


@dataclass
class GeoReference:
    """
    Reference point for local ENU coordinate system.
    
    Attributes:
        lat_ref: Reference latitude in degrees
        lon_ref: Reference longitude in degrees
        alt_ref: Reference altitude in meters (above WGS84 ellipsoid)
        lat_ref_rad: Reference latitude in radians
        lon_ref_rad: Reference longitude in radians
        meters_per_deg_lat: Meters per degree latitude at reference
        meters_per_deg_lon: Meters per degree longitude at reference
    """
    lat_ref: float
    lon_ref: float
    alt_ref: float
    lat_ref_rad: float
    lon_ref_rad: float
    meters_per_deg_lat: float
    meters_per_deg_lon: float


def create_geo_reference(lat: float, lon: float, alt: float = 0.0) -> GeoReference:
    """
    Create a geographic reference point for ENU conversions.
    
    Args:
        lat: Reference latitude in degrees
        lon: Reference longitude in degrees
        alt: Reference altitude in meters (default: 0.0)
        
    Returns:
        GeoReference object with precomputed conversion factors
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Calculate meters per degree at this latitude
    # Using the WGS84 ellipsoid for accuracy
    
    # Radius of curvature in the prime vertical (N)
    sin_lat = math.sin(lat_rad)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    
    # Radius of curvature in the meridian (M)
    M = WGS84_A * (1 - WGS84_E2) / ((1 - WGS84_E2 * sin_lat * sin_lat) ** 1.5)
    
    # Meters per degree
    meters_per_deg_lat = M * math.pi / 180.0
    meters_per_deg_lon = N * math.cos(lat_rad) * math.pi / 180.0
    
    return GeoReference(
        lat_ref=lat,
        lon_ref=lon,
        alt_ref=alt,
        lat_ref_rad=lat_rad,
        lon_ref_rad=lon_rad,
        meters_per_deg_lat=meters_per_deg_lat,
        meters_per_deg_lon=meters_per_deg_lon
    )


def gnss_to_enu(
    lat: float,
    lon: float,
    alt: float,
    ref: GeoReference
) -> Tuple[float, float, float]:
    """
    Convert GNSS lat/lon/alt to local ENU (East-North-Up) coordinates.
    
    Uses equirectangular projection which is accurate for small distances
    (< 100 km from reference).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters (above WGS84 ellipsoid)
        ref: GeoReference object defining the local origin
        
    Returns:
        Tuple of (east, north, up) in meters relative to reference
    """
    # Calculate differences in degrees
    dlat = lat - ref.lat_ref
    dlon = lon - ref.lon_ref
    dalt = alt - ref.alt_ref
    
    # Convert to meters using precomputed factors
    east = dlon * ref.meters_per_deg_lon
    north = dlat * ref.meters_per_deg_lat
    up = dalt
    
    return (east, north, up)


def gnss_to_enu_array(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    ref: GeoReference
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized version of gnss_to_enu for numpy arrays.
    
    Args:
        lat: Array of latitudes in degrees
        lon: Array of longitudes in degrees
        alt: Array of altitudes in meters
        ref: GeoReference object defining the local origin
        
    Returns:
        Tuple of (east, north, up) arrays in meters
    """
    dlat = lat - ref.lat_ref
    dlon = lon - ref.lon_ref
    dalt = alt - ref.alt_ref
    
    east = dlon * ref.meters_per_deg_lon
    north = dlat * ref.meters_per_deg_lat
    up = dalt
    
    return (east, north, up)


def enu_to_gnss(
    east: float,
    north: float,
    up: float,
    ref: GeoReference
) -> Tuple[float, float, float]:
    """
    Convert local ENU coordinates back to GNSS lat/lon/alt.
    
    Args:
        east: East coordinate in meters
        north: North coordinate in meters
        up: Up coordinate in meters
        ref: GeoReference object defining the local origin
        
    Returns:
        Tuple of (latitude, longitude, altitude) in degrees/meters
    """
    dlon = east / ref.meters_per_deg_lon
    dlat = north / ref.meters_per_deg_lat
    
    lat = ref.lat_ref + dlat
    lon = ref.lon_ref + dlon
    alt = ref.alt_ref + up
    
    return (lat, lon, alt)


class GNSSToLocalConverter:
    """
    Stateful converter that establishes reference on first sample.
    
    Usage:
        converter = GNSSToLocalConverter()
        
        # First call establishes the reference point
        x, y, z = converter.convert(lat0, lon0, alt0)  # Returns (0, 0, 0)
        
        # Subsequent calls return ENU relative to first point
        x, y, z = converter.convert(lat1, lon1, alt1)  # Returns (east, north, up)
    """
    
    def __init__(self):
        """Initialize converter without a reference (set on first sample)."""
        self.reference: Optional[GeoReference] = None
        self.is_initialized = False
        
    def reset(self):
        """Reset the converter to uninitialized state."""
        self.reference = None
        self.is_initialized = False
        
    def set_reference(self, lat: float, lon: float, alt: float = 0.0):
        """
        Manually set the reference point.
        
        Args:
            lat: Reference latitude in degrees
            lon: Reference longitude in degrees
            alt: Reference altitude in meters
        """
        self.reference = create_geo_reference(lat, lon, alt)
        self.is_initialized = True
        
    def convert(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        """
        Convert GNSS coordinates to local ENU meters.
        
        On first call, establishes the reference point and returns (0, 0, 0).
        Subsequent calls return position relative to the first sample.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
            
        Returns:
            Tuple of (east, north, up) in meters
        """
        if not self.is_initialized:
            self.reference = create_geo_reference(lat, lon, alt)
            self.is_initialized = True
            return (0.0, 0.0, 0.0)
            
        return gnss_to_enu(lat, lon, alt, self.reference)
    
    def convert_array(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert arrays of GNSS coordinates to local ENU.
        
        Uses first element as reference if not already initialized.
        
        Args:
            lat: Array of latitudes in degrees
            lon: Array of longitudes in degrees
            alt: Array of altitudes in meters
            
        Returns:
            Tuple of (east, north, up) arrays in meters
        """
        if len(lat) == 0:
            return (np.array([]), np.array([]), np.array([]))
            
        if not self.is_initialized:
            self.reference = create_geo_reference(lat[0], lon[0], alt[0])
            self.is_initialized = True
            
        return gnss_to_enu_array(lat, lon, alt, self.reference)
    
    def get_reference_info(self) -> Optional[dict]:
        """
        Get information about the current reference point.
        
        Returns:
            Dictionary with reference info, or None if not initialized
        """
        if not self.is_initialized or self.reference is None:
            return None
            
        return {
            'lat_deg': self.reference.lat_ref,
            'lon_deg': self.reference.lon_ref,
            'alt_m': self.reference.alt_ref,
            'meters_per_deg_lat': self.reference.meters_per_deg_lat,
            'meters_per_deg_lon': self.reference.meters_per_deg_lon
        }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point in degrees
        lat2, lon2: Second point in degrees
        
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's mean radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


# Sanity check / self-test
def _run_sanity_checks():
    """Run basic sanity checks on the conversion functions."""
    print("Running GNSS to ENU conversion sanity checks...")
    
    # Test 1: Reference point should map to (0, 0, 0)
    ref = create_geo_reference(40.0, -74.0, 100.0)
    e, n, u = gnss_to_enu(40.0, -74.0, 100.0, ref)
    assert abs(e) < 1e-10, f"Reference E should be 0, got {e}"
    assert abs(n) < 1e-10, f"Reference N should be 0, got {n}"
    assert abs(u) < 1e-10, f"Reference U should be 0, got {u}"
    print("  [PASS] Reference point maps to (0, 0, 0)")
    
    # Test 2: Moving 1 degree north should be ~111 km
    e, n, u = gnss_to_enu(41.0, -74.0, 100.0, ref)
    expected_north = 111000  # Approximately 111 km
    assert 110000 < n < 112000, f"1 deg north should be ~111km, got {n/1000:.1f}km"
    print(f"  [PASS] 1 deg north = {n/1000:.1f} km (expected ~111 km)")
    
    # Test 3: Moving 1 degree east at lat 40 should be ~85 km
    e, n, u = gnss_to_enu(40.0, -73.0, 100.0, ref)
    expected_east = 85000  # Approximately 85 km at 40 deg lat
    assert 84000 < e < 86000, f"1 deg east at lat 40 should be ~85km, got {e/1000:.1f}km"
    print(f"  [PASS] 1 deg east at lat 40 = {e/1000:.1f} km (expected ~85 km)")
    
    # Test 4: Altitude change
    e, n, u = gnss_to_enu(40.0, -74.0, 200.0, ref)
    assert abs(u - 100.0) < 1e-10, f"100m altitude change should give U=100, got {u}"
    print("  [PASS] Altitude change maps correctly")
    
    # Test 5: Converter class
    converter = GNSSToLocalConverter()
    x, y, z = converter.convert(40.0, -74.0, 100.0)
    assert (x, y, z) == (0.0, 0.0, 0.0), "First conversion should be (0, 0, 0)"
    print("  [PASS] Converter initializes on first sample")
    
    x, y, z = converter.convert(40.001, -74.001, 100.0)
    assert abs(x) > 0 and abs(y) > 0, "Subsequent conversions should have non-zero values"
    print(f"  [PASS] Subsequent conversion: ({x:.2f}, {y:.2f}, {z:.2f}) m")
    
    # Test 6: Round-trip conversion
    lat_orig, lon_orig, alt_orig = 40.5, -74.5, 150.0
    ref2 = create_geo_reference(40.0, -74.0, 100.0)
    e, n, u = gnss_to_enu(lat_orig, lon_orig, alt_orig, ref2)
    lat_back, lon_back, alt_back = enu_to_gnss(e, n, u, ref2)
    assert abs(lat_back - lat_orig) < 1e-9, f"Latitude round-trip failed"
    assert abs(lon_back - lon_orig) < 1e-9, f"Longitude round-trip failed"
    assert abs(alt_back - alt_orig) < 1e-9, f"Altitude round-trip failed"
    print("  [PASS] Round-trip conversion accurate")
    
    print("All sanity checks passed!")
    return True


if __name__ == "__main__":
    _run_sanity_checks()

