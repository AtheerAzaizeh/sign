"""
opentrend/harvesters/base.py
Base classes for the Global Fashion Brain Harvester System

This module defines the abstract base class that all fashion data
harvesters must inherit from, plus common utilities.

Plugin Architecture:
    All harvesters inherit from FashionSource and implement:
    - harvest() -> Generator of FashionItem
    - validate() -> bool
    
    The GlobalHarvester manager runs all plugins in parallel threads.
"""

from __future__ import annotations

import abc
import time
import random
import sqlite3
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Generator, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FashionItem:
    """Universal data structure for all harvested fashion content."""
    source: str                          # e.g., "reddit", "vogue", "unsplash"
    source_url: str                      # Original URL
    image_url: Optional[str] = None      # Direct image URL if available
    local_path: Optional[str] = None     # Local saved path
    title: str = ""                      # Description/title
    content: str = ""                    # Full text content
    tags: List[str] = field(default_factory=list)  # Original tags
    predicted_labels: List[str] = field(default_factory=list)  # AI-predicted
    category: str = ""                   # Fashion category
    engagement: int = 0                  # Likes, upvotes, etc.
    timestamp: str = ""                  # When harvested
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def content_hash(self) -> str:
        """Generate unique hash for deduplication."""
        content = f"{self.source}:{self.source_url}:{self.title}"
        return hashlib.md5(content.encode()).hexdigest()


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class FashionSource(abc.ABC):
    """
    Abstract base class for all fashion data harvesters.
    
    All source plugins must implement:
    - name: str property
    - harvest() -> Generator[FashionItem, None, None]
    - validate() -> bool
    
    Optional to override:
    - setup() -> Called before harvesting
    - cleanup() -> Called after harvesting
    """
    
    # User-Agent rotation pool
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
    ]
    
    def __init__(self, data_dir: str = "./data/harvested"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.items_harvested = 0
        self.errors = 0
        self.enabled = True
        self._setup_session()
    
    def _setup_session(self):
        """Configure requests session with headers."""
        self.session.headers.update({
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def rotate_user_agent(self):
        """Switch to a different User-Agent."""
        self.session.headers['User-Agent'] = random.choice(self.USER_AGENTS)
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this source."""
        pass
    
    @abc.abstractmethod
    def harvest(self) -> Generator[FashionItem, None, None]:
        """
        Main harvesting method. Yields FashionItem objects.
        Must be implemented by all subclasses.
        """
        pass
    
    @abc.abstractmethod
    def validate(self) -> bool:
        """
        Check if the source is accessible and properly configured.
        Returns True if ready to harvest.
        """
        pass
    
    def setup(self):
        """Optional setup before harvesting. Override if needed."""
        pass
    
    def cleanup(self):
        """Optional cleanup after harvesting. Override if needed."""
        pass
    
    def rate_limit(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """Add random delay between requests."""
        time.sleep(random.uniform(min_delay, max_delay))
    
    def download_image(self, url: str, filename: str = None) -> Optional[str]:
        """Download image and return local path."""
        try:
            if not filename:
                ext = url.split('.')[-1].split('?')[0][:4]
                filename = f"{hashlib.md5(url.encode()).hexdigest()}.{ext}"
            
            local_path = self.data_dir / self.name / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            return str(local_path)
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return None
    
    def get_logger(self) -> logging.Logger:
        """Get source-specific logger."""
        return logging.getLogger(f"harvester.{self.name}")


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class HarvestDatabase:
    """
    SQLite database for storing harvested fashion data.
    Thread-safe with connection pooling.
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS fashion_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content_hash TEXT UNIQUE,
        source TEXT NOT NULL,
        source_url TEXT NOT NULL,
        image_url TEXT,
        local_path TEXT,
        title TEXT,
        content TEXT,
        tags TEXT,
        predicted_labels TEXT,
        category TEXT,
        engagement INTEGER DEFAULT 0,
        harvested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed BOOLEAN DEFAULT FALSE,
        metadata TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_source ON fashion_items(source);
    CREATE INDEX IF NOT EXISTS idx_processed ON fashion_items(processed);
    CREATE INDEX IF NOT EXISTS idx_harvested_at ON fashion_items(harvested_at);
    
    CREATE TABLE IF NOT EXISTS harvest_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        items_harvested INTEGER DEFAULT 0,
        errors INTEGER DEFAULT 0,
        started_at TIMESTAMP,
        completed_at TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS training_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        new_samples INTEGER,
        previous_accuracy REAL,
        new_accuracy REAL,
        model_version TEXT
    );
    """
    
    def __init__(self, db_path: str = "./data/fashion_brain.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def save_item(self, item: FashionItem) -> bool:
        """Save a fashion item to database. Returns True if new."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR IGNORE INTO fashion_items (
                            content_hash, source, source_url, image_url,
                            local_path, title, content, tags, predicted_labels,
                            category, engagement, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        item.content_hash,
                        item.source,
                        item.source_url,
                        item.image_url,
                        item.local_path,
                        item.title,
                        item.content,
                        ','.join(item.tags),
                        ','.join(item.predicted_labels),
                        item.category,
                        item.engagement,
                        str(item.metadata)
                    ))
                    return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Database error: {e}")
                return False
    
    def get_unprocessed_count(self) -> int:
        """Get count of items not yet processed by auto-labeler."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fashion_items WHERE processed = FALSE")
            return cursor.fetchone()[0]
    
    def get_unprocessed_items(self, limit: int = 100) -> List[Dict]:
        """Get unprocessed items for labeling."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM fashion_items 
                WHERE processed = FALSE AND local_path IS NOT NULL
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def mark_processed(self, item_ids: List[int], labels: Dict[int, List[str]]):
        """Mark items as processed and save predicted labels."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for item_id in item_ids:
                    item_labels = labels.get(item_id, [])
                    cursor.execute("""
                        UPDATE fashion_items 
                        SET processed = TRUE, predicted_labels = ?
                        WHERE id = ?
                    """, (','.join(item_labels), item_id))
    
    def get_total_items(self) -> int:
        """Get total items in database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fashion_items")
            return cursor.fetchone()[0]
    
    def get_stats(self) -> Dict:
        """Get harvesting statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total by source
            cursor.execute("""
                SELECT source, COUNT(*) as count, SUM(engagement) as total_engagement
                FROM fashion_items GROUP BY source
            """)
            source_stats = {row[0]: {'count': row[1], 'engagement': row[2]} 
                          for row in cursor.fetchall()}
            
            # Processing stats
            cursor.execute("SELECT COUNT(*) FROM fashion_items WHERE processed = TRUE")
            processed = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM fashion_items")
            total = cursor.fetchone()[0]
            
            return {
                'total_items': total,
                'processed': processed,
                'pending': total - processed,
                'by_source': source_stats
            }
    
    def log_training(self, new_samples: int, prev_acc: float, new_acc: float, version: str):
        """Log training event."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO training_log (new_samples, previous_accuracy, new_accuracy, model_version)
                VALUES (?, ?, ?, ?)
            """, (new_samples, prev_acc, new_acc, version))


# =============================================================================
# GLOBAL HARVESTER MANAGER
# =============================================================================

class GlobalHarvester:
    """
    Main orchestrator for all fashion data harvesters.
    
    Features:
    - Parallel execution of all source plugins
    - Automatic error isolation (one failing source doesn't stop others)
    - Continuous harvesting mode with configurable intervals
    - Statistics and monitoring
    
    Usage:
        harvester = GlobalHarvester()
        harvester.register(RedditHarvester())
        harvester.register(VogueHarvester())
        harvester.run_once()  # Or harvester.run_continuous()
    """
    
    def __init__(
        self,
        db_path: str = "./data/fashion_brain.db",
        max_workers: int = 4,
        harvest_interval: int = 3600  # 1 hour
    ):
        self.db = HarvestDatabase(db_path)
        self.sources: Dict[str, FashionSource] = {}
        self.max_workers = max_workers
        self.harvest_interval = harvest_interval
        self.is_running = False
        self._lock = Lock()
        
        logger.info("GlobalHarvester initialized")
    
    def register(self, source: FashionSource):
        """Register a harvester source plugin."""
        self.sources[source.name] = source
        logger.info(f"Registered source: {source.name}")
    
    def unregister(self, source_name: str):
        """Remove a source plugin."""
        if source_name in self.sources:
            del self.sources[source_name]
            logger.info(f"Unregistered source: {source_name}")
    
    def _harvest_source(self, source: FashionSource) -> Dict:
        """Harvest from a single source. Run in thread."""
        stats = {
            'source': source.name,
            'items': 0,
            'errors': 0,
            'started_at': datetime.now().isoformat()
        }
        
        try:
            # Validate source
            if not source.validate():
                logger.warning(f"Source {source.name} validation failed, skipping")
                stats['errors'] = 1
                return stats
            
            # Setup
            source.setup()
            
            # Harvest
            for item in source.harvest():
                try:
                    if self.db.save_item(item):
                        stats['items'] += 1
                except Exception as e:
                    logger.error(f"Error saving item from {source.name}: {e}")
                    stats['errors'] += 1
            
            # Cleanup
            source.cleanup()
            
        except Exception as e:
            logger.error(f"Critical error harvesting {source.name}: {e}")
            stats['errors'] += 1
        
        stats['completed_at'] = datetime.now().isoformat()
        return stats
    
    def run_once(self) -> List[Dict]:
        """Run all harvesters once in parallel."""
        logger.info(f"Starting harvest cycle with {len(self.sources)} sources")
        
        enabled_sources = [s for s in self.sources.values() if s.enabled]
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._harvest_source, source): source.name 
                for source in enabled_sources
            }
            
            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {source_name}: {result['items']} items, {result['errors']} errors")
                except Exception as e:
                    logger.error(f"Source {source_name} failed: {e}")
                    results.append({'source': source_name, 'items': 0, 'errors': 1})
        
        # Summary
        total_items = sum(r['items'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        logger.info(f"Harvest cycle complete: {total_items} items, {total_errors} errors")
        
        return results
    
    def run_continuous(self):
        """Run harvesting continuously with interval."""
        self.is_running = True
        logger.info(f"Starting continuous harvesting (interval: {self.harvest_interval}s)")
        
        while self.is_running:
            try:
                self.run_once()
                logger.info(f"Sleeping for {self.harvest_interval}s...")
                time.sleep(self.harvest_interval)
            except KeyboardInterrupt:
                logger.info("Stopping continuous harvesting...")
                self.is_running = False
            except Exception as e:
                logger.error(f"Error in continuous mode: {e}")
                time.sleep(60)  # Wait 1 min on error
    
    def stop(self):
        """Stop continuous harvesting."""
        self.is_running = False
    
    def get_stats(self) -> Dict:
        """Get overall statistics."""
        db_stats = self.db.get_stats()
        db_stats['registered_sources'] = list(self.sources.keys())
        db_stats['enabled_sources'] = [s.name for s in self.sources.values() if s.enabled]
        return db_stats
    
    def print_status(self):
        """Print current status."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("🧠 GLOBAL FASHION BRAIN - STATUS")
        print("="*60)
        print(f"Total Items: {stats['total_items']}")
        print(f"Processed: {stats['processed']}")
        print(f"Pending: {stats['pending']}")
        print(f"\n📊 By Source:")
        for source, data in stats.get('by_source', {}).items():
            print(f"  {source}: {data['count']} items")
        print(f"\n🔌 Registered Sources: {', '.join(stats['registered_sources'])}")
        print("="*60)
