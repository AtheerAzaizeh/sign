"""
opentrend/trend_scraper.py
Fashion Trend Data Acquisition Module

This module provides ethical web scraping capabilities to build a timestamped
fashion image dataset for trend forecasting. It scrapes publicly available
fashion galleries and organizes images with date/designer/category metadata.

⚠️  ETHICAL SCRAPING DISCLAIMER
================================
This code is provided for EDUCATIONAL purposes only. Before scraping any website:

1. CHECK robots.txt: Always visit {website}/robots.txt first
2. RESPECT rate limits: Use delays between requests (2-5 seconds minimum)
3. READ Terms of Service: Many sites prohibit automated scraping
4. USE APIs when available: Prefer official APIs over scraping
5. CACHE responses: Don't re-download the same content
6. IDENTIFY yourself: Use descriptive User-Agent strings
7. STOP if blocked: Don't bypass rate limiting or CAPTCHAs

LEGAL ALTERNATIVES:
- Unsplash API (fashion tag): https://unsplash.com/developers
- Pexels API: https://www.pexels.com/api/
- Pinterest API (requires approval): https://developers.pinterest.com/
- Kaggle Fashion Datasets: https://www.kaggle.com/datasets?tags=11403-Fashion

The author is not responsible for misuse of this code.
================================

Recommended Data Sources (Updated 2025):
=========================================

1. **Unsplash API** (RECOMMENDED - Ethical)
   - Free API with fashion category
   - High-quality, timestamped images
   - Easy to use, no scraping needed
   - https://unsplash.com/developers

2. **Google Trends + Stock Photos**
   - Use pytrends for time-series data
   - Download representative images from stock sites
   - Combine for timestamped dataset

3. **Fashion Kaggle Datasets** (Static but large)
   - Fashion Product Images: 44K items with metadata
   - DeepFashion: 800K items (academic use)
   - iMaterialist: Competition dataset

4. **Social Media APIs** (Requires approval)
   - Instagram Graph API (Business accounts)
   - Pinterest API (Apply for access)
   - TikTok Research API

Usage:
    scraper = FashionTrendScraper(output_dir="./data/raw")
    
    # Ethical: Use Unsplash API
    scraper.download_from_unsplash("puffer jacket", count=50)
    
    # Or: Download from curated Kaggle dataset
    scraper.download_kaggle_dataset("paramaggarwal/fashion-product-images-dataset")
"""

from __future__ import annotations

import os
import re
import json
import time
import random
import hashlib
import logging
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse, quote

import requests
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not installed. HTML parsing disabled.")

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("selenium not installed. JavaScript rendering disabled.")


# =============================================================================
# CONSTANTS
# =============================================================================

# Ethical User-Agent identifying the bot
USER_AGENT = (
    "OpenTrendBot/1.0 (Fashion Research; Educational Use; "
    "+https://github.com/opentrend) Python/3.x"
)

# Request headers mimicking a browser
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Minimum delay between requests (seconds)
MIN_DELAY = 2.0
MAX_DELAY = 5.0

# Fashion categories for organization
FASHION_CATEGORIES = [
    "dress", "jacket", "coat", "blazer", "sweater", "hoodie",
    "pants", "jeans", "skirt", "shorts", "top", "blouse",
    "shirt", "t-shirt", "suit", "jumpsuit", "romper",
    "activewear", "swimwear", "lingerie", "outerwear",
    "knitwear", "denim", "leather", "accessories"
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ScrapedImage:
    """Represents a scraped fashion image with metadata."""
    url: str
    local_path: Optional[str] = None
    source: str = "unknown"
    date: Optional[datetime] = None
    designer: Optional[str] = None
    category: Optional[str] = None
    season: Optional[str] = None
    collection: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def generate_filename(self) -> str:
        """Generate a descriptive filename from metadata."""
        parts = []
        
        # Date
        if self.date:
            parts.append(self.date.strftime("%Y-%m"))
        else:
            parts.append(datetime.now().strftime("%Y-%m"))
        
        # Designer/Source
        if self.designer:
            safe_designer = re.sub(r'[^\w\-]', '_', self.designer)[:30]
            parts.append(safe_designer.lower())
        elif self.source:
            parts.append(self.source[:20])
        
        # Category
        if self.category:
            parts.append(self.category.lower())
        
        # Season
        if self.season:
            parts.append(self.season.lower().replace(" ", "_"))
        
        # Hash for uniqueness
        url_hash = hashlib.md5(self.url.encode()).hexdigest()[:8]
        parts.append(url_hash)
        
        return "_".join(parts) + ".jpg"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "url": self.url,
            "local_path": self.local_path,
            "source": self.source,
            "date": self.date.isoformat() if self.date else None,
            "designer": self.designer,
            "category": self.category,
            "season": self.season,
            "collection": self.collection,
            "tags": self.tags
        }


# =============================================================================
# MAIN SCRAPER CLASS
# =============================================================================

class FashionTrendScraper:
    """
    Ethical fashion image scraper for building timestamped datasets.
    
    ⚠️ Always check robots.txt and Terms of Service before scraping!
    
    This class provides multiple methods to acquire fashion images:
    1. Unsplash API (Recommended - fully legal)
    2. Pexels API (Free, requires API key)
    3. Curated Kaggle downloads
    4. Web scraping (Use responsibly!)
    
    Usage:
        scraper = FashionTrendScraper(output_dir="./data/raw")
        
        # Method 1: Unsplash API (Best)
        scraper.download_from_unsplash("puffer jacket", count=50)
        
        # Method 2: Organize existing images
        scraper.organize_dataset("./downloads", "./data/organized")
    """
    
    def __init__(
        self,
        output_dir: str = "./data/raw",
        min_delay: float = MIN_DELAY,
        max_delay: float = MAX_DELAY,
        unsplash_api_key: str = None,
        pexels_api_key: str = None
    ):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Root directory for saved images
            min_delay: Minimum seconds between requests
            max_delay: Maximum seconds between requests
            unsplash_api_key: Optional Unsplash API key
            pexels_api_key: Optional Pexels API key
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_delay = min_delay
        self.max_delay = max_delay
        
        self.unsplash_api_key = unsplash_api_key or os.getenv("UNSPLASH_API_KEY")
        self.pexels_api_key = pexels_api_key or os.getenv("PEXELS_API_KEY")
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        
        # Downloaded images metadata
        self.images: List[ScrapedImage] = []
        
        logger.info(f"FashionTrendScraper initialized. Output: {self.output_dir}")
    
    def _delay(self):
        """Add random delay between requests."""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)
    
    def _download_image(self, url: str, save_path: Path) -> bool:
        """Download a single image."""
        try:
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Validate it's an image
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                logger.warning(f"Not an image: {url}")
                return False
            
            # Save image
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return False
    
    # =========================================================================
    # METHOD 1: UNSPLASH API (RECOMMENDED)
    # =========================================================================
    
    def download_from_unsplash(
        self,
        query: str,
        count: int = 30,
        collection_name: str = None
    ) -> List[ScrapedImage]:
        """
        Download images from Unsplash API (free, ethical, high-quality).
        
        Get your API key at: https://unsplash.com/developers
        
        Args:
            query: Search term (e.g., "puffer jacket", "street style")
            count: Number of images to download (max 30 per request)
            collection_name: Optional collection folder name
            
        Returns:
            List of ScrapedImage objects
        """
        if not self.unsplash_api_key:
            logger.warning(
                "Unsplash API key not set. Get one at https://unsplash.com/developers\n"
                "Set via: export UNSPLASH_API_KEY=your_key_here"
            )
            return []
        
        logger.info(f"📷 Downloading from Unsplash: '{query}' ({count} images)")
        
        # Create collection folder
        safe_query = re.sub(r'[^\w\-]', '_', query)
        date_str = datetime.now().strftime("%Y-%m")
        
        if collection_name:
            folder_name = f"{date_str}_{collection_name}"
        else:
            folder_name = f"{date_str}_{safe_query}"
        
        save_dir = self.output_dir / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # API request
        url = "https://api.unsplash.com/search/photos"
        params = {
            "query": query,
            "per_page": min(count, 30),
            "orientation": "portrait",
            "content_filter": "high"
        }
        headers = {
            "Authorization": f"Client-ID {self.unsplash_api_key}"
        }
        
        try:
            response = self.session.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Unsplash API error: {e}")
            return []
        
        results = data.get("results", [])
        scraped_images = []
        
        for i, photo in enumerate(tqdm(results, desc=f"Downloading {query}")):
            # Extract metadata
            img_url = photo.get("urls", {}).get("regular", "")
            created_at = photo.get("created_at", "")
            user = photo.get("user", {}).get("name", "unknown")
            description = photo.get("description") or photo.get("alt_description") or ""
            tags = [t.get("title", "") for t in photo.get("tags", [])]
            
            # Parse date
            try:
                date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except:
                date = datetime.now()
            
            # Detect category from tags/description
            category = self._detect_category(description + " " + " ".join(tags))
            
            # Create ScrapedImage
            scraped = ScrapedImage(
                url=img_url,
                source="unsplash",
                date=date,
                designer=user,
                category=category,
                tags=tags[:5]
            )
            
            # Download
            filename = scraped.generate_filename()
            save_path = save_dir / filename
            
            if self._download_image(img_url, save_path):
                scraped.local_path = str(save_path)
                scraped_images.append(scraped)
            
            self._delay()
        
        # Save metadata
        self._save_metadata(scraped_images, save_dir)
        
        logger.info(f"✓ Downloaded {len(scraped_images)} images to {save_dir}")
        self.images.extend(scraped_images)
        
        return scraped_images
    
    # =========================================================================
    # METHOD 2: PEXELS API
    # =========================================================================
    
    def download_from_pexels(
        self,
        query: str,
        count: int = 30,
        collection_name: str = None
    ) -> List[ScrapedImage]:
        """
        Download images from Pexels API (free, requires key).
        
        Get your API key at: https://www.pexels.com/api/
        
        Args:
            query: Search term
            count: Number of images
            collection_name: Optional folder name
            
        Returns:
            List of ScrapedImage objects
        """
        if not self.pexels_api_key:
            logger.warning(
                "Pexels API key not set. Get one at https://www.pexels.com/api/\n"
                "Set via: export PEXELS_API_KEY=your_key_here"
            )
            return []
        
        logger.info(f"📷 Downloading from Pexels: '{query}' ({count} images)")
        
        # Create folder
        safe_query = re.sub(r'[^\w\-]', '_', query)
        date_str = datetime.now().strftime("%Y-%m")
        folder_name = collection_name or f"{date_str}_{safe_query}"
        
        save_dir = self.output_dir / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # API request
        url = "https://api.pexels.com/v1/search"
        params = {"query": query, "per_page": min(count, 80), "orientation": "portrait"}
        headers = {"Authorization": self.pexels_api_key}
        
        try:
            response = self.session.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Pexels API error: {e}")
            return []
        
        photos = data.get("photos", [])
        scraped_images = []
        
        for photo in tqdm(photos[:count], desc=f"Downloading {query}"):
            img_url = photo.get("src", {}).get("large", "")
            photographer = photo.get("photographer", "unknown")
            alt = photo.get("alt", "")
            
            category = self._detect_category(alt)
            
            scraped = ScrapedImage(
                url=img_url,
                source="pexels",
                date=datetime.now(),
                designer=photographer,
                category=category
            )
            
            filename = scraped.generate_filename()
            save_path = save_dir / filename
            
            if self._download_image(img_url, save_path):
                scraped.local_path = str(save_path)
                scraped_images.append(scraped)
            
            self._delay()
        
        self._save_metadata(scraped_images, save_dir)
        self.images.extend(scraped_images)
        
        return scraped_images
    
    # =========================================================================
    # METHOD 3: KAGGLE DATASET DOWNLOAD
    # =========================================================================
    
    def download_kaggle_dataset(
        self,
        dataset_id: str = "paramaggarwal/fashion-product-images-dataset"
    ) -> Path:
        """
        Download a fashion dataset from Kaggle.
        
        Requires kaggle CLI: pip install kaggle
        And API credentials in ~/.kaggle/kaggle.json
        
        Recommended datasets:
        - paramaggarwal/fashion-product-images-dataset (44K images)
        - mozhikchiranjeeb/fashion-clothing-products-image-processing
        - agrigorev/clothing-dataset-full
        
        Args:
            dataset_id: Kaggle dataset identifier
            
        Returns:
            Path to downloaded dataset
        """
        import subprocess
        
        logger.info(f"📦 Downloading Kaggle dataset: {dataset_id}")
        
        save_dir = self.output_dir / "kaggle" / dataset_id.split("/")[-1]
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            cmd = [
                "kaggle", "datasets", "download",
                "-d", dataset_id,
                "-p", str(save_dir),
                "--unzip"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ Downloaded to {save_dir}")
                return save_dir
            else:
                logger.error(f"Kaggle download failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            logger.error(
                "kaggle CLI not found. Install with: pip install kaggle\n"
                "Configure API key: https://www.kaggle.com/docs/api"
            )
            return None
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _detect_category(self, text: str) -> Optional[str]:
        """Detect fashion category from text."""
        text_lower = text.lower()
        
        for category in FASHION_CATEGORIES:
            if category in text_lower:
                return category
        
        return None
    
    def _save_metadata(self, images: List[ScrapedImage], save_dir: Path):
        """Save image metadata as JSON."""
        metadata_path = save_dir / "metadata.json"
        
        data = {
            "scraped_at": datetime.now().isoformat(),
            "count": len(images),
            "images": [img.to_dict() for img in images]
        }
        
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved metadata to {metadata_path}")
    
    # =========================================================================
    # DATASET ORGANIZATION
    # =========================================================================
    
    def organize_for_opentrend(
        self,
        source_dir: str,
        output_dir: str = None
    ) -> Path:
        """
        Organize images into OpenTrend-compatible folder structure.
        
        Reads metadata.json and organizes images into:
            {output_dir}/{date}_{collection}/{category}/image.jpg
        
        This structure allows FashionDataLoader to use folder names as labels.
        
        Args:
            source_dir: Directory with scraped images and metadata.json
            output_dir: Output directory (default: ./data/organized)
            
        Returns:
            Path to organized dataset
        """
        source = Path(source_dir)
        output = Path(output_dir) if output_dir else self.output_dir / "organized"
        output.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Organizing dataset for OpenTrend...")
        
        # Load metadata
        metadata_path = source / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"No metadata.json found in {source}")
            return self._organize_by_folder(source, output)
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        organized_count = 0
        
        for img_data in metadata.get("images", []):
            local_path = img_data.get("local_path")
            if not local_path or not Path(local_path).exists():
                continue
            
            # Determine category folder
            category = img_data.get("category") or "uncategorized"
            
            # Get date prefix
            date_str = img_data.get("date", "")[:7] or datetime.now().strftime("%Y-%m")
            
            # Create category folder
            category_dir = output / category.lower().replace(" ", "_")
            category_dir.mkdir(exist_ok=True)
            
            # Copy/move image
            src = Path(local_path)
            dst = category_dir / src.name
            
            import shutil
            shutil.copy2(src, dst)
            organized_count += 1
        
        logger.info(f"✓ Organized {organized_count} images to {output}")
        return output
    
    def _organize_by_folder(self, source: Path, output: Path) -> Path:
        """Organize images by parent folder name as category."""
        import shutil
        
        count = 0
        for img_path in source.rglob("*.jpg"):
            category = img_path.parent.name
            dst_dir = output / category
            dst_dir.mkdir(exist_ok=True)
            
            shutil.copy2(img_path, dst_dir / img_path.name)
            count += 1
        
        logger.info(f"Organized {count} images by folder structure")
        return output


# =============================================================================
# SAMPLE GOOGLE TRENDS INTEGRATION
# =============================================================================

def create_timestamped_dataset(
    queries: List[str],
    scraper: FashionTrendScraper,
    images_per_query: int = 20
) -> Dict[str, List[ScrapedImage]]:
    """
    Create a timestamped dataset by combining image scraping with Google Trends.
    
    This function:
    1. Downloads images for each fashion query
    2. Augments with Google Trends data for time-series
    
    Args:
        queries: List of fashion search terms
        scraper: FashionTrendScraper instance
        images_per_query: Images to download per query
        
    Returns:
        Dict mapping query -> list of images
    """
    from data_loader import FashionDataLoader
    
    loader = FashionDataLoader()
    results = {}
    
    for query in queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {query}")
        logger.info(f"{'='*50}")
        
        # Download images
        images = scraper.download_from_unsplash(query, count=images_per_query)
        
        # Get trend data
        trend_df = loader.ensure_trend_history(query, years=3)
        
        # Log trend info
        if not trend_df.empty:
            latest = trend_df['y'].iloc[-1]
            growth = ((trend_df['y'].iloc[-1] - trend_df['y'].iloc[0]) / 
                     max(trend_df['y'].iloc[0], 1)) * 100
            logger.info(f"  Trend: {latest:.0f}/100 ({growth:+.1f}% growth)")
        
        results[query] = images
    
    return results


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          OPENTREND FASHION DATA ACQUISITION                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ⚠️  ETHICAL SCRAPING NOTICE                                          ║
║  Always check robots.txt and Terms of Service before scraping.        ║
║  This tool prioritizes API-based methods (Unsplash, Pexels) that      ║
║  are fully legal and ethical.                                          ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize scraper
    scraper = FashionTrendScraper(output_dir="./data/raw")
    
    # Demo: Download from Unsplash (requires API key)
    print("\n📋 To use the Unsplash API:")
    print("   1. Sign up at https://unsplash.com/developers")
    print("   2. Create an app to get your Access Key")
    print("   3. export UNSPLASH_API_KEY=your_access_key")
    print("   4. Run: scraper.download_from_unsplash('puffer jacket', count=30)")
    
    print("\n📋 To download Kaggle datasets:")
    print("   1. pip install kaggle")
    print("   2. Configure ~/.kaggle/kaggle.json")
    print("   3. Run: scraper.download_kaggle_dataset('paramaggarwal/fashion-product-images-dataset')")
    
    print("\n📋 Dataset Structure for OpenTrend:")
    print("""
    data/
    ├── raw/                          # Scraped images
    │   ├── 2025-01_puffer_jacket/
    │   │   ├── 2025-01_designer_jacket_abc123.jpg
    │   │   └── metadata.json
    │   └── 2025-01_cargo_pants/
    │       └── ...
    │
    └── organized/                    # For FashionDataLoader
        ├── jacket/                   # Category as folder name
        │   ├── img001.jpg
        │   └── img002.jpg
        ├── pants/
        └── dress/
    """)
    
    print("\n✅ Scraper module ready!")
    print("   Import: from trend_scraper import FashionTrendScraper")
