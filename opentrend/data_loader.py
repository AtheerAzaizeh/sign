"""
opentrend/data_loader.py
Production-Grade Fashion Dataset Loader with Google Trends Integration

This module provides:
1. Loading fashion image datasets (DeepFashion, Kaggle, custom directories)
2. Ground-truth label extraction from folder structure
3. Google Trends time-series generation with caching and retry logic
4. Exponential backoff for API rate limiting

Dataset Download Instructions:
==============================

Option 1: DeepFashion (Large-scale, requires registration)
    Website: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
    Structure after extraction:
        data/deepfashion/Img/{category}/{subcategory}/{image}.jpg

Option 2: Kaggle Fashion Product Images (Easy, ~15GB)
    Command: kaggle datasets download -d paramaggarwal/fashion-product-images-dataset
    Structure after extraction:
        data/fashion_products/images/{id}.jpg
        data/fashion_products/styles.csv

Option 3: Custom Dataset
    Organize your images as:
        data/custom/{category_label}/{image}.jpg
    Example:
        data/custom/puffer_jacket/img001.jpg
        data/custom/denim_jacket/img002.jpg
"""

from __future__ import annotations

import json
import hashlib
import time
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Iterator, Any
from functools import wraps

import pandas as pd
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional tqdm import with fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback tqdm that just returns the iterable."""
        return iterable

# Import existing OpenTrend modules
try:
    from signals.trend_tracker import TrendTracker
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from signals.trend_tracker import TrendTracker
    except ImportError:
        TrendTracker = None
        logger.warning("TrendTracker not available. Google Trends features disabled.")


# =============================================================================
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# =============================================================================

def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple = (Exception,)
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential calculation
        jitter: Add random jitter to prevent thundering herd
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter (±25%)
                    if jitter:
                        delay *= (0.75 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FashionItem:
    """
    Represents a single fashion item from the dataset.
    
    Attributes:
        image_path: Absolute path to the image file
        ground_truth_label: Category label from folder structure (e.g., "denim_jacket")
        color: Optional detected/extracted color
        attributes: Additional metadata dictionary
        timestamp: Optional timestamp for time-series datasets
    """
    image_path: str
    ground_truth_label: str
    color: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Normalize the ground truth label."""
        # Convert to lowercase, replace underscores with spaces for display
        self.ground_truth_label = self.ground_truth_label.lower().strip()
    
    @property
    def search_query(self) -> str:
        """
        Generate a Google Trends search query.
        Converts "denim_jacket" -> "denim jacket" for better search results.
        """
        query = self.ground_truth_label.replace('_', ' ').replace('-', ' ')
        if self.color:
            query = f"{self.color} {query}"
        return query
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'image_path': self.image_path,
            'ground_truth_label': self.ground_truth_label,
            'color': self.color,
            'search_query': self.search_query,
            'attributes': self.attributes,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass 
class FashionDataset:
    """
    Container for a collection of fashion items.
    
    Attributes:
        items: List of FashionItem objects
        source: Dataset source identifier
        loaded_at: Timestamp when dataset was loaded
    """
    items: List[FashionItem]
    source: str
    loaded_at: datetime = field(default_factory=datetime.now)
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self) -> Iterator[FashionItem]:
        return iter(self.items)
    
    def __getitem__(self, idx: int) -> FashionItem:
        return self.items[idx]
    
    @property
    def labels(self) -> List[str]:
        """Get all ground truth labels."""
        return [item.ground_truth_label for item in self.items]
    
    @property
    def unique_labels(self) -> List[str]:
        """Get unique ground truth labels."""
        return list(set(self.labels))
    
    @property
    def image_paths(self) -> List[str]:
        """Get all image paths."""
        return [item.image_path for item in self.items]
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get count of items per label."""
        from collections import Counter
        return dict(Counter(self.labels))
    
    def filter_by_label(self, label: str) -> 'FashionDataset':
        """Return new dataset filtered to a specific label."""
        filtered = [item for item in self.items 
                   if item.ground_truth_label == label.lower()]
        return FashionDataset(items=filtered, source=self.source)
    
    def sample(self, n: int, random_state: int = 42) -> 'FashionDataset':
        """Return a random sample of the dataset."""
        np.random.seed(random_state)
        indices = np.random.choice(len(self.items), min(n, len(self.items)), replace=False)
        sampled = [self.items[i] for i in indices]
        return FashionDataset(items=sampled, source=f"{self.source}_sample")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([item.to_dict() for item in self.items])


# =============================================================================
# TREND HISTORY CACHE
# =============================================================================

class TrendCache:
    """
    File-based cache for Google Trends data.
    
    Stores trend data as JSON files to prevent redundant API calls.
    Cache entries expire after a configurable TTL.
    """
    
    def __init__(self, cache_dir: Path, ttl_days: int = 7):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_days: Time-to-live for cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_days * 24 * 3600
        self.index_file = self.cache_dir / "_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except json.JSONDecodeError:
                self.index = {}
        else:
            self.index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_cache_key(self, query: str, years: int) -> str:
        """Generate unique cache key for a query."""
        raw = f"{query.lower().strip()}_{years}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{cache_key}.csv"
    
    def is_valid(self, query: str, years: int) -> bool:
        """Check if cache entry exists and is not expired."""
        cache_key = self._get_cache_key(query, years)
        
        if cache_key not in self.index:
            return False
        
        # Check TTL
        cached_at = self.index[cache_key].get('cached_at', 0)
        age = time.time() - cached_at
        
        if age > self.ttl_seconds:
            logger.debug(f"Cache expired for '{query}' (age: {age/3600:.1f}h)")
            return False
        
        # Check file exists
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return False
        
        return True
    
    def get(self, query: str, years: int) -> Optional[pd.DataFrame]:
        """
        Retrieve cached trend data.
        
        Returns None if not cached or expired.
        """
        if not self.is_valid(query, years):
            return None
        
        cache_key = self._get_cache_key(query, years)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            df = pd.read_csv(cache_path, parse_dates=['ds'])
            logger.info(f"✓ Cache hit for '{query}' ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Cache read error for '{query}': {e}")
            return None
    
    def set(self, query: str, years: int, df: pd.DataFrame):
        """Store trend data in cache."""
        cache_key = self._get_cache_key(query, years)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            df.to_csv(cache_path, index=False)
            
            self.index[cache_key] = {
                'query': query,
                'years': years,
                'rows': len(df),
                'cached_at': time.time()
            }
            self._save_index()
            
            logger.debug(f"Cached '{query}' ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Cache write error for '{query}': {e}")
    
    def clear(self, query: str = None):
        """Clear cache entries. If query is None, clear all."""
        if query is None:
            # Clear all
            for cache_key in list(self.index.keys()):
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
            self.index = {}
            self._save_index()
            logger.info("Cache cleared")
        else:
            # Clear specific query (all years)
            keys_to_remove = [k for k, v in self.index.items() if v.get('query') == query]
            for cache_key in keys_to_remove:
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                del self.index[cache_key]
            self._save_index()


# =============================================================================
# MAIN DATA LOADER
# =============================================================================

class FashionDataLoader:
    """
    Production-grade loader for fashion datasets with Google Trends integration.
    
    Features:
    - Load from various dataset formats (DeepFashion, Kaggle, custom)
    - Extract ground-truth labels from folder structure
    - Generate time-series via Google Trends with caching
    - Exponential backoff for API rate limiting
    
    Usage:
        loader = FashionDataLoader()
        
        # Load dataset
        dataset = loader.load_dataset("./data/fashion/{category}/{image}.jpg")
        
        # Ensure trend history for an item (with caching)
        df = loader.ensure_trend_history("puffer_jacket", years=5)
    """
    
    # Supported image extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    
    def __init__(
        self,
        cache_dir: str = None,
        cache_ttl_days: int = 7,
        api_delay: float = 2.0
    ):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory for caching trend data (default: ./data/.trends_cache)
            cache_ttl_days: Cache time-to-live in days
            api_delay: Delay between Google Trends API calls
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "data" / ".trends_cache"
        
        self.cache = TrendCache(Path(cache_dir), ttl_days=cache_ttl_days)
        self.api_delay = api_delay
        self._trend_tracker = None
    
    @property
    def trend_tracker(self) -> Optional['TrendTracker']:
        """Lazy initialization of TrendTracker."""
        if self._trend_tracker is None and TrendTracker is not None:
            self._trend_tracker = TrendTracker(retries=3, backoff_factor=1.5)
        return self._trend_tracker
    
    # =========================================================================
    # DATASET LOADING
    # =========================================================================
    
    def load_dataset(
        self,
        path: str,
        max_items: int = None,
        show_progress: bool = True
    ) -> FashionDataset:
        """
        Load a fashion dataset from a directory.
        
        Expected structure:
            path/
                {category1}/
                    image1.jpg
                    image2.jpg
                {category2}/
                    image3.jpg
        
        The folder name becomes the "Ground Truth Label" for images inside.
        
        Args:
            path: Root directory path
            max_items: Maximum items to load (for testing)
            show_progress: Show tqdm progress bar
            
        Returns:
            FashionDataset with loaded items
            
        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If no valid images found
        """
        root = Path(path)
        
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root}")
        
        # Find all image files
        image_files = []
        for ext in self.VALID_EXTENSIONS:
            image_files.extend(root.rglob(f"*{ext}"))
            image_files.extend(root.rglob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No valid images found in {root}")
        
        # Apply limit
        if max_items:
            image_files = image_files[:max_items]
        
        # Load items with progress bar
        items = []
        iterator = tqdm(image_files, desc="Loading images", disable=not show_progress)
        
        for img_path in iterator:
            # Extract ground truth label from parent folder
            relative = img_path.relative_to(root)
            parts = relative.parts
            
            if len(parts) >= 2:
                # Has category folder
                ground_truth_label = parts[0]
            else:
                # No category folder, use filename
                ground_truth_label = img_path.stem
            
            item = FashionItem(
                image_path=str(img_path.absolute()),
                ground_truth_label=ground_truth_label
            )
            items.append(item)
        
        dataset = FashionDataset(items=items, source=str(root))
        
        logger.info(f"✓ Loaded {len(dataset)} items from {root}")
        logger.info(f"  Labels: {len(dataset.unique_labels)} unique categories")
        
        return dataset
    
    def load_kaggle_fashion(
        self,
        path: str,
        max_items: int = None,
        show_progress: bool = True
    ) -> FashionDataset:
        """
        Load the Kaggle Fashion Product Images dataset.
        
        Uses styles.csv for rich metadata including articleType as label.
        
        Args:
            path: Path to extracted dataset
            max_items: Maximum items to load
            show_progress: Show progress bar
            
        Returns:
            FashionDataset with metadata from CSV
        """
        root = Path(path)
        images_dir = root / "images"
        styles_csv = root / "styles.csv"
        
        if not styles_csv.exists():
            logger.warning("styles.csv not found, falling back to directory load")
            return self.load_dataset(str(images_dir), max_items, show_progress)
        
        # Load metadata
        try:
            df = pd.read_csv(styles_csv, on_bad_lines='skip')
        except Exception as e:
            logger.error(f"Error reading styles.csv: {e}")
            return self.load_dataset(str(images_dir), max_items, show_progress)
        
        if max_items:
            df = df.head(max_items)
        
        items = []
        iterator = tqdm(df.iterrows(), total=len(df), desc="Loading Kaggle dataset", 
                       disable=not show_progress)
        
        for _, row in iterator:
            img_id = row.get('id', '')
            img_path = images_dir / f"{img_id}.jpg"
            
            if not img_path.exists():
                continue
            
            # Use articleType as ground truth label
            label = str(row.get('articleType', row.get('masterCategory', 'unknown')))
            
            item = FashionItem(
                image_path=str(img_path.absolute()),
                ground_truth_label=label,
                color=str(row.get('baseColour', '')).lower() if pd.notna(row.get('baseColour')) else None,
                attributes={
                    'gender': row.get('gender'),
                    'season': row.get('season'),
                    'year': row.get('year'),
                    'usage': row.get('usage'),
                    'productDisplayName': row.get('productDisplayName')
                }
            )
            items.append(item)
        
        dataset = FashionDataset(items=items, source='kaggle_fashion_products')
        logger.info(f"✓ Loaded {len(dataset)} items from Kaggle dataset")
        
        return dataset
    
    # =========================================================================
    # GOOGLE TRENDS TIME-SERIES GENERATION
    # =========================================================================
    
    def ensure_trend_history(
        self,
        item_label: str,
        years: int = 5,
        geo: str = ''
    ) -> pd.DataFrame:
        """
        Ensure trend history exists for an item label.
        
        This is the KEY METHOD for generating time-series data for static image datasets.
        
        Logic:
        1. Check cache first
        2. If not cached, fetch from Google Trends with retry/backoff
        3. Cache the result
        4. Return DataFrame suitable for Prophet
        
        Args:
            item_label: Fashion item label (e.g., "denim_jacket" or "puffer jacket")
            years: Years of historical data to fetch
            geo: Geographic region ('' = worldwide)
            
        Returns:
            DataFrame with columns:
                - ds: datetime (Prophet-compatible)
                - y: interest value (0-100)
        """
        # Normalize label to search query
        search_query = item_label.replace('_', ' ').replace('-', ' ').strip().lower()
        
        # Check cache
        cached_df = self.cache.get(search_query, years)
        if cached_df is not None:
            return cached_df
        
        # Fetch from Google Trends with retry
        logger.info(f"📡 Fetching Google Trends for '{search_query}' ({years} years)...")
        
        df = self._fetch_trends_with_retry(search_query, years, geo)
        
        # Cache result
        self.cache.set(search_query, years, df)
        
        # Apply rate limiting delay
        time.sleep(self.api_delay)
        
        return df
    
    @retry_with_backoff(
        max_retries=5,
        base_delay=5.0,
        max_delay=120.0,
        exponential_base=2.0,
        jitter=True
    )
    def _fetch_trends_with_retry(
        self,
        query: str,
        years: int,
        geo: str
    ) -> pd.DataFrame:
        """
        Fetch Google Trends data with exponential backoff retry.
        
        This method is decorated with retry logic to handle rate limiting.
        """
        if self.trend_tracker is None:
            logger.warning("TrendTracker not available. Generating synthetic data.")
            return self._generate_synthetic_trends(query, years)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            
            timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
            
            # Fetch using TrendTracker
            df = self.trend_tracker.get_interest_over_time(
                keyword=query,
                timeframe=timeframe,
                geo=geo
            )
            
            if df.empty:
                logger.warning(f"No Google Trends data for '{query}'. Using synthetic.")
                return self._generate_synthetic_trends(query, years)
            
            # Rename columns for Prophet
            df = df.rename(columns={'date': 'ds', 'interest': 'y'})
            df['ds'] = pd.to_datetime(df['ds'])
            
            logger.info(f"✓ Fetched {len(df)} data points for '{query}'")
            return df
            
        except Exception as e:
            logger.error(f"Google Trends API error: {e}")
            raise
    
    def _generate_synthetic_trends(
        self,
        query: str,
        years: int
    ) -> pd.DataFrame:
        """
        Generate synthetic trend data as fallback.
        
        Creates realistic seasonal data when API is unavailable.
        """
        np.random.seed(hash(query) % (2**32))
        
        periods = years * 52  # Weekly data
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='W')
        
        t = np.arange(periods)
        
        # Base level + slight trend
        base_level = 50 + np.random.uniform(-10, 10)
        trend = np.linspace(0, np.random.uniform(-10, 15), periods)
        
        # Yearly seasonality (fashion is seasonal)
        seasonality = 15 * np.sin(2 * np.pi * t / 52)
        
        # Random noise
        noise = np.random.normal(0, 5, periods)
        
        values = np.clip(base_level + trend + seasonality + noise, 0, 100)
        
        df = pd.DataFrame({'ds': dates, 'y': values})
        
        logger.info(f"Generated synthetic trend data for '{query}' ({len(df)} points)")
        return df
    
    def ensure_trends_for_dataset(
        self,
        dataset: FashionDataset,
        years: int = 5,
        sample_labels: int = None,
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Ensure trend history for all unique labels in a dataset.
        
        Args:
            dataset: FashionDataset to process
            years: Years of historical data
            sample_labels: Limit to N most common labels (to avoid rate limiting)
            show_progress: Show progress bar
            
        Returns:
            Dict mapping label -> trend DataFrame
        """
        # Get label distribution
        label_counts = dataset.get_label_distribution()
        
        # Sort by frequency (most common first)
        sorted_labels = sorted(label_counts.keys(), key=lambda x: -label_counts[x])
        
        if sample_labels:
            sorted_labels = sorted_labels[:sample_labels]
        
        logger.info(f"📈 Fetching trends for {len(sorted_labels)} labels...")
        
        trends = {}
        iterator = tqdm(sorted_labels, desc="Fetching trends", disable=not show_progress)
        
        for label in iterator:
            iterator.set_postfix_str(label[:20])
            
            try:
                df = self.ensure_trend_history(label, years=years)
                trends[label] = df
            except Exception as e:
                logger.error(f"Failed to fetch trends for '{label}': {e}")
        
        return trends


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧥 OPENTREND DATA LOADER - Production Demo")
    print("=" * 70)
    
    loader = FashionDataLoader()
    
    # Demo: Generate trend history for fashion items
    print("\n📊 Fetching Google Trends Data (with caching)")
    print("-" * 50)
    
    fashion_items = [
        "puffer_jacket",
        "cargo_pants", 
        "floral_dress",
        "denim_jacket"
    ]
    
    for item in fashion_items:
        df = loader.ensure_trend_history(item, years=1)
        
        if not df.empty:
            latest = df['y'].iloc[-1]
            mean = df['y'].mean()
            trend = "📈" if df['y'].iloc[-1] > df['y'].iloc[0] else "📉"
            
            print(f"\n  {item.replace('_', ' ').title()}: {trend}")
            print(f"    Latest: {latest:.0f}/100 | Avg: {mean:.1f}/100 | Points: {len(df)}")
    
    print("\n" + "=" * 70)
    print("✅ Data Loader Ready")
    print("=" * 70)
    print("""
    Usage:
        loader = FashionDataLoader()
        
        # Load dataset
        dataset = loader.load_dataset("./data/fashion")
        
        # Get trend history (with caching + retry)
        df = loader.ensure_trend_history("puffer_jacket", years=5)
    """)
