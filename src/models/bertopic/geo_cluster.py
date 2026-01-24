"""
Geospatial Clustering for Venues.

Uses HDBSCAN to cluster venues by geographic location,
creating region identifiers for context-aware recommendations.
"""
import logging
from typing import Optional, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import BERTopicConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class RegionInfo:
    """Information about a geographic region."""
    region_id: int
    name: str
    center_lat: float
    center_lon: float
    venue_count: int
    radius_km: float


class GeoClusterer:
    """Cluster venues by geographic location."""

    # Earth's radius in km
    EARTH_RADIUS_KM = 6371.0

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        cluster_selection_epsilon: float = 0.5,
    ):
        """
        Initialize the geo clusterer.

        Args:
            min_cluster_size: Minimum venues per cluster.
            min_samples: Minimum samples for core points.
            cluster_selection_epsilon: Distance threshold for cluster merging (km).
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

        self.clusterer = None
        self.labels = None
        self.regions = None

    def _haversine_distance(
        self,
        lat1: np.ndarray,
        lon1: np.ndarray,
        lat2: np.ndarray,
        lon2: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate haversine distance between coordinate pairs.

        Args:
            lat1, lon1: First coordinates (radians).
            lat2, lon2: Second coordinates (radians).

        Returns:
            Distances in kilometers.
        """
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return self.EARTH_RADIUS_KM * c

    def _coords_to_cartesian(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        """
        Convert lat/lon to 3D Cartesian coordinates.

        This provides better distance preservation for clustering.
        """
        lat_rad = np.radians(latitudes)
        lon_rad = np.radians(longitudes)

        x = self.EARTH_RADIUS_KM * np.cos(lat_rad) * np.cos(lon_rad)
        y = self.EARTH_RADIUS_KM * np.cos(lat_rad) * np.sin(lon_rad)
        z = self.EARTH_RADIUS_KM * np.sin(lat_rad)

        return np.column_stack([x, y, z])

    def fit(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        venue_ids: Optional[list] = None,
    ) -> np.ndarray:
        """
        Fit the clusterer on venue coordinates.

        Args:
            latitudes: Array of latitudes.
            longitudes: Array of longitudes.
            venue_ids: Optional venue identifiers.

        Returns:
            Cluster labels for each venue.
        """
        from hdbscan import HDBSCAN

        logger.info(f"Clustering {len(latitudes)} venues by location")

        # Convert to Cartesian for better clustering
        coords_3d = self._coords_to_cartesian(latitudes, longitudes)

        # Create and fit HDBSCAN
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric="euclidean",
            prediction_data=True,
        )

        self.labels = self.clusterer.fit_predict(coords_3d)

        # Generate region information
        self.regions = self._generate_region_info(
            latitudes, longitudes, venue_ids
        )

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = (self.labels == -1).sum()
        logger.info(f"Found {n_clusters} regions, {n_noise} outliers")

        return self.labels

    def _generate_region_info(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        venue_ids: Optional[list] = None,
    ) -> list[RegionInfo]:
        """Generate information about each region."""
        regions = []

        unique_labels = sorted(set(self.labels) - {-1})

        for label in unique_labels:
            mask = self.labels == label

            # Calculate center
            center_lat = latitudes[mask].mean()
            center_lon = longitudes[mask].mean()

            # Calculate radius (max distance from center)
            center_lat_rad = np.radians(center_lat)
            center_lon_rad = np.radians(center_lon)
            lats_rad = np.radians(latitudes[mask])
            lons_rad = np.radians(longitudes[mask])

            distances = self._haversine_distance(
                center_lat_rad, center_lon_rad,
                lats_rad, lons_rad,
            )
            radius = distances.max()

            # Generate region name based on location
            region_name = self._generate_region_name(center_lat, center_lon, label)

            regions.append(RegionInfo(
                region_id=label,
                name=region_name,
                center_lat=center_lat,
                center_lon=center_lon,
                venue_count=mask.sum(),
                radius_km=radius,
            ))

        return regions

    def _generate_region_name(
        self,
        lat: float,
        lon: float,
        label: int,
    ) -> str:
        """
        Generate a descriptive name for a region.

        In a production system, this would use reverse geocoding.
        """
        # Simple directional naming
        lat_dir = "North" if lat > 0 else "South"
        lon_dir = "East" if lon > 0 else "West"

        return f"Region_{label}_{lat_dir}{lon_dir}"

    def predict(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        """
        Predict region labels for new venues.

        Args:
            latitudes: Array of latitudes.
            longitudes: Array of longitudes.

        Returns:
            Predicted region labels.
        """
        if self.clusterer is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")

        from hdbscan import approximate_predict

        coords_3d = self._coords_to_cartesian(latitudes, longitudes)
        labels, _ = approximate_predict(self.clusterer, coords_3d)

        return labels

    def get_region_embeddings(self, embedding_dim: int = 32) -> np.ndarray:
        """
        Generate embeddings for each region.

        Uses center coordinates and radius to create feature vectors.
        """
        if self.regions is None:
            raise ValueError("No regions available. Call fit() first.")

        embeddings = []

        for region in self.regions:
            # Create feature vector from region properties
            features = [
                np.sin(np.radians(region.center_lat)),
                np.cos(np.radians(region.center_lat)),
                np.sin(np.radians(region.center_lon)),
                np.cos(np.radians(region.center_lon)),
                np.log1p(region.radius_km),
                np.log1p(region.venue_count),
            ]

            # Pad or project to target dimension
            if len(features) < embedding_dim:
                features.extend([0.0] * (embedding_dim - len(features)))
            else:
                features = features[:embedding_dim]

            embeddings.append(features)

        return np.array(embeddings)

    def get_venue_region_matrix(self, n_venues: int) -> np.ndarray:
        """
        Get venue-region assignment matrix.

        Returns:
            Binary matrix [n_venues, n_regions].
        """
        if self.labels is None:
            raise ValueError("No labels available. Call fit() first.")

        n_regions = len(self.regions)
        matrix = np.zeros((n_venues, n_regions))

        for i, label in enumerate(self.labels):
            if label >= 0:
                matrix[i, label] = 1

        return matrix

    def to_dataframe(self, venue_ids: list) -> pd.DataFrame:
        """Convert clustering results to DataFrame."""
        if self.labels is None:
            raise ValueError("No labels available. Call fit() first.")

        region_names = {r.region_id: r.name for r in self.regions}
        region_names[-1] = "Outlier"

        df = pd.DataFrame({
            "venue_id": venue_ids,
            "region_id": self.labels,
            "region_name": [region_names.get(l, "Unknown") for l in self.labels],
        })

        return df


class TimeContextEncoder:
    """Encode temporal context for venues."""

    # Time period definitions
    TIME_PERIODS = {
        "morning": (6, 11),
        "afternoon": (11, 17),
        "evening": (17, 21),
        "night": (21, 6),
    }

    SEASONS = {
        "spring": (3, 5),
        "summer": (6, 8),
        "fall": (9, 11),
        "winter": (12, 2),
    }

    DAY_TYPES = ["weekday", "weekend"]

    def __init__(self, embedding_dim: int = 16):
        """
        Initialize the time context encoder.

        Args:
            embedding_dim: Dimension of time embeddings.
        """
        self.embedding_dim = embedding_dim

        # Create embeddings for each time feature
        n_periods = len(self.TIME_PERIODS)
        n_seasons = len(self.SEASONS)
        n_day_types = len(self.DAY_TYPES)

        np.random.seed(42)
        self.period_embeddings = np.random.randn(n_periods, embedding_dim) * 0.1
        self.season_embeddings = np.random.randn(n_seasons, embedding_dim) * 0.1
        self.day_type_embeddings = np.random.randn(n_day_types, embedding_dim) * 0.1

    def encode(
        self,
        hours: np.ndarray,
        months: np.ndarray,
        is_weekend: np.ndarray,
    ) -> np.ndarray:
        """
        Encode temporal context.

        Args:
            hours: Hour of day (0-23).
            months: Month of year (1-12).
            is_weekend: Whether it's a weekend.

        Returns:
            Time embeddings [n_samples, embedding_dim].
        """
        n_samples = len(hours)
        embeddings = np.zeros((n_samples, self.embedding_dim))

        for i in range(n_samples):
            # Get period embedding
            period_idx = self._get_period_idx(hours[i])
            embeddings[i] += self.period_embeddings[period_idx]

            # Get season embedding
            season_idx = self._get_season_idx(months[i])
            embeddings[i] += self.season_embeddings[season_idx]

            # Get day type embedding
            day_idx = 1 if is_weekend[i] else 0
            embeddings[i] += self.day_type_embeddings[day_idx]

        return embeddings

    def _get_period_idx(self, hour: int) -> int:
        """Get time period index from hour."""
        for i, (name, (start, end)) in enumerate(self.TIME_PERIODS.items()):
            if name == "night":
                if hour >= start or hour < end:
                    return i
            elif start <= hour < end:
                return i
        return 0

    def _get_season_idx(self, month: int) -> int:
        """Get season index from month."""
        for i, (name, (start, end)) in enumerate(self.SEASONS.items()):
            if name == "winter":
                if month >= start or month <= end:
                    return i
            elif start <= month <= end:
                return i
        return 0


class WeatherContextEncoder:
    """Encode weather context for venues."""

    WEATHER_TYPES = [
        "sunny", "cloudy", "rainy", "snowy", "windy", "foggy"
    ]

    TEMPERATURE_BINS = [
        "freezing",  # < 0°C
        "cold",      # 0-10°C
        "mild",      # 10-20°C
        "warm",      # 20-30°C
        "hot",       # > 30°C
    ]

    def __init__(self, embedding_dim: int = 16):
        """
        Initialize the weather context encoder.

        Args:
            embedding_dim: Dimension of weather embeddings.
        """
        self.embedding_dim = embedding_dim

        np.random.seed(43)
        self.weather_embeddings = np.random.randn(
            len(self.WEATHER_TYPES), embedding_dim
        ) * 0.1
        self.temp_embeddings = np.random.randn(
            len(self.TEMPERATURE_BINS), embedding_dim
        ) * 0.1

    def encode(
        self,
        weather_types: list[str],
        temperatures: np.ndarray,
    ) -> np.ndarray:
        """
        Encode weather context.

        Args:
            weather_types: List of weather type strings.
            temperatures: Temperature values in Celsius.

        Returns:
            Weather embeddings [n_samples, embedding_dim].
        """
        n_samples = len(weather_types)
        embeddings = np.zeros((n_samples, self.embedding_dim))

        for i in range(n_samples):
            # Get weather type embedding
            weather_idx = self._get_weather_idx(weather_types[i])
            embeddings[i] += self.weather_embeddings[weather_idx]

            # Get temperature embedding
            temp_idx = self._get_temp_idx(temperatures[i])
            embeddings[i] += self.temp_embeddings[temp_idx]

        return embeddings

    def _get_weather_idx(self, weather: str) -> int:
        """Get weather type index."""
        weather = weather.lower()
        for i, wtype in enumerate(self.WEATHER_TYPES):
            if wtype in weather:
                return i
        return 0  # Default to sunny

    def _get_temp_idx(self, temp: float) -> int:
        """Get temperature bin index."""
        if temp < 0:
            return 0
        elif temp < 10:
            return 1
        elif temp < 20:
            return 2
        elif temp < 30:
            return 3
        else:
            return 4


def cluster_venues_by_location(
    businesses_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: str = "business_id",
    min_cluster_size: int = 10,
    save_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, GeoClusterer]:
    """
    Cluster venues by geographic location.

    Args:
        businesses_df: DataFrame with venue data.
        lat_col: Column name for latitude.
        lon_col: Column name for longitude.
        id_col: Column name for venue ID.
        min_cluster_size: Minimum venues per cluster.
        save_path: Path to save results.

    Returns:
        Tuple of (venue_regions DataFrame, clusterer).
    """
    # Extract coordinates
    latitudes = businesses_df[lat_col].values
    longitudes = businesses_df[lon_col].values
    venue_ids = businesses_df[id_col].tolist()

    # Cluster
    clusterer = GeoClusterer(min_cluster_size=min_cluster_size)
    clusterer.fit(latitudes, longitudes, venue_ids)

    # Convert to DataFrame
    result = clusterer.to_dataframe(venue_ids)

    # Add center coordinates
    region_centers = {r.region_id: (r.center_lat, r.center_lon) for r in clusterer.regions}
    result["region_center_lat"] = result["region_id"].map(
        lambda x: region_centers.get(x, (0, 0))[0]
    )
    result["region_center_lon"] = result["region_id"].map(
        lambda x: region_centers.get(x, (0, 0))[1]
    )

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(save_path, index=False)
        logger.info(f"Saved venue regions to {save_path}")

    return result, clusterer
