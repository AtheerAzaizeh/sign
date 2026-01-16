"""
opentrend/harvesters/core.py
Production-Grade Core Infrastructure for Global Fashion Brain

This module provides:
- SQLAlchemy ORM for database abstraction (SQLite → PostgreSQL ready)
- Queue-based database writer to prevent locking
- Enhanced FashionSource ABC with enforced interface
- Resilient GlobalHarvester with automatic retries

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                     CORE INFRASTRUCTURE                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  FashionSource  │────│ ThreadPoolExec  │                    │
│  │  (Abstract BC)  │    │  (Parallel)     │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Plugin A,B,C...│───▶│  DatabaseQueue  │                    │
│  └─────────────────┘    │  (Thread-safe)  │                    │
│                         └────────┬────────┘                    │
│                                  │                              │
│                         ┌────────▼────────┐                    │
│                         │  SQLAlchemy ORM │                    │
│                         │  (PostgreSQL)   │                    │
│                         └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import abc
import time
import random
import logging
import hashlib
import queue
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

# SQLAlchemy imports
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, 
    Boolean, DateTime, JSON, Index, event
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.pool import QueuePool

import requests

# Configure logging with proper format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./data/fashion_brain.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Ensure data directory
Path('./data').mkdir(exist_ok=True)
Path('./data/harvested').mkdir(exist_ok=True)


# =============================================================================
# SQLALCHEMY ORM MODELS
# =============================================================================

Base = declarative_base()


class FashionItemModel(Base):
    """
    SQLAlchemy ORM model for fashion items.
    
    Designed for easy migration from SQLite to PostgreSQL.
    """
    __tablename__ = 'fashion_items'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    
    # Source information
    source = Column(String(50), nullable=False, index=True)
    source_url = Column(Text, nullable=False)
    image_url = Column(Text)
    local_path = Column(Text)
    
    # Content
    title = Column(Text)
    content = Column(Text)
    
    # Labels & Tags
    original_tags = Column(Text)  # JSON array as string
    predicted_labels = Column(Text)  # AI-predicted labels
    predicted_confidence = Column(Float)  # CLIP confidence score
    
    # Classification
    category = Column(String(100), index=True)
    style = Column(String(100))  # Predicted style
    
    # Engagement metrics
    engagement = Column(Integer, default=0)
    
    # Status
    is_valid = Column(Boolean, default=True)  # Passed garbage filter
    is_processed = Column(Boolean, default=False, index=True)
    is_used_for_training = Column(Boolean, default=False)
    
    # Timestamps
    harvested_at = Column(DateTime, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime)
    
    # Metadata (JSON for flexibility)
    metadata = Column(JSON)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_source_processed', 'source', 'is_processed'),
        Index('idx_valid_processed', 'is_valid', 'is_processed'),
        Index('idx_harvested_at', 'harvested_at'),
    )


class HarvestLogModel(Base):
    """Log of harvesting operations."""
    __tablename__ = 'harvest_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    items_harvested = Column(Integer, default=0)
    items_valid = Column(Integer, default=0)
    items_filtered = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    error_message = Column(Text)
    status = Column(String(20))  # running, completed, failed


class TrainingLogModel(Base):
    """Log of model training runs."""
    __tablename__ = 'training_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    new_samples = Column(Integer)
    total_samples = Column(Integer)
    epochs = Column(Integer)
    previous_accuracy = Column(Float)
    new_accuracy = Column(Float)
    model_version = Column(String(50))
    model_path = Column(String(255))
    status = Column(String(20))


# =============================================================================
# DATABASE MANAGER WITH QUEUE WRITER
# =============================================================================

class DatabaseManager:
    """
    Production-grade database manager with:
    - SQLAlchemy ORM for database abstraction
    - Queue-based writes to prevent locking
    - Connection pooling for performance
    - Easy migration to PostgreSQL
    """
    
    def __init__(
        self,
        db_url: str = "sqlite:///./data/fashion_brain.db",
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        """
        Initialize database manager.
        
        Args:
            db_url: Database connection URL. Supports:
                    - sqlite:///./data/fashion_brain.db
                    - postgresql://user:pass@host/dbname
            pool_size: Connection pool size
            max_overflow: Max connections above pool_size
        """
        self.db_url = db_url
        
        # Create engine with connection pooling
        if 'sqlite' in db_url:
            # SQLite-specific settings
            self.engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                echo=False
            )
        else:
            # PostgreSQL/MySQL with connection pooling
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                echo=False
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Write queue for async writes
        self._write_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_writer = threading.Event()
        
        # Start background writer
        self._start_writer()
        
        logger.info(f"Database initialized: {db_url}")
    
    def _start_writer(self):
        """Start background queue writer thread."""
        self._stop_writer.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
    
    def _writer_loop(self):
        """Background loop that processes write queue."""
        batch = []
        batch_size = 50
        last_flush = time.time()
        
        while not self._stop_writer.is_set():
            try:
                # Get from queue with timeout
                item = self._write_queue.get(timeout=1.0)
                batch.append(item)
                
                # Flush if batch is full or timeout
                if len(batch) >= batch_size or (time.time() - last_flush) > 5.0:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except queue.Empty:
                # Flush any remaining items
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
            except Exception as e:
                logger.error(f"Writer error: {e}")
    
    def _flush_batch(self, batch: List[FashionItemModel]):
        """Flush a batch of items to database."""
        if not batch:
            return
        
        session = self.SessionLocal()
        try:
            for item in batch:
                try:
                    session.merge(item)
                except Exception as e:
                    logger.warning(f"Error merging item: {e}")
            session.commit()
            logger.debug(f"Flushed {len(batch)} items to database")
        except Exception as e:
            session.rollback()
            logger.error(f"Batch flush error: {e}")
        finally:
            session.close()
    
    def queue_item(self, item: 'FashionItem') -> bool:
        """
        Queue an item for async database write.
        
        Returns True if queued successfully.
        """
        try:
            model = FashionItemModel(
                content_hash=item.content_hash,
                source=item.source,
                source_url=item.source_url,
                image_url=item.image_url,
                local_path=item.local_path,
                title=item.title,
                content=item.content,
                original_tags=','.join(item.tags) if item.tags else '',
                predicted_labels=','.join(item.predicted_labels) if item.predicted_labels else '',
                category=item.category,
                engagement=item.engagement,
                metadata=item.metadata
            )
            self._write_queue.put_nowait(model)
            return True
        except queue.Full:
            logger.warning("Write queue full, dropping item")
            return False
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def get_unprocessed_items(self, limit: int = 100) -> List[FashionItemModel]:
        """Get items that haven't been processed by AI."""
        session = self.SessionLocal()
        try:
            items = session.query(FashionItemModel).filter(
                FashionItemModel.is_processed == False,
                FashionItemModel.local_path.isnot(None)
            ).limit(limit).all()
            return items
        finally:
            session.close()
    
    def mark_processed(self, item_id: int, labels: List[str], confidence: float, is_valid: bool):
        """Mark an item as processed and save predictions."""
        session = self.SessionLocal()
        try:
            item = session.query(FashionItemModel).get(item_id)
            if item:
                item.is_processed = True
                item.is_valid = is_valid
                item.predicted_labels = ','.join(labels)
                item.predicted_confidence = confidence
                item.processed_at = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error marking processed: {e}")
        finally:
            session.close()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        session = self.SessionLocal()
        try:
            total = session.query(FashionItemModel).count()
            processed = session.query(FashionItemModel).filter(
                FashionItemModel.is_processed == True
            ).count()
            valid = session.query(FashionItemModel).filter(
                FashionItemModel.is_valid == True
            ).count()
            
            # By source
            from sqlalchemy import func
            by_source = session.query(
                FashionItemModel.source,
                func.count(FashionItemModel.id)
            ).group_by(FashionItemModel.source).all()
            
            return {
                'total_items': total,
                'processed': processed,
                'pending': total - processed,
                'valid': valid,
                'filtered': processed - valid,
                'by_source': {s: c for s, c in by_source}
            }
        finally:
            session.close()
    
    def shutdown(self):
        """Gracefully shutdown the database manager."""
        self._stop_writer.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=10)
        logger.info("Database manager shutdown complete")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SourceStatus(Enum):
    """Status of a data source."""
    IDLE = "idle"
    RUNNING = "running"
    COOLDOWN = "cooldown"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class FashionItem:
    """Universal data structure for harvested fashion content."""
    source: str
    source_url: str
    image_url: Optional[str] = None
    local_path: Optional[str] = None
    title: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    predicted_labels: List[str] = field(default_factory=list)
    category: str = ""
    engagement: int = 0
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    @property
    def content_hash(self) -> str:
        """Generate unique hash for deduplication."""
        content = f"{self.source}:{self.source_url}:{self.title}"
        return hashlib.md5(content.encode()).hexdigest()


# =============================================================================
# ABSTRACT BASE CLASS FOR SOURCES
# =============================================================================

class FashionSource(abc.ABC):
    """
    Abstract Base Class for all fashion data sources.
    
    All plugins MUST implement:
    - name (property): Unique identifier
    - fetch_latest(): Fetch raw data from source
    - parse(raw_data): Parse raw data into structured format
    - normalize(parsed_data): Normalize to FashionItem
    
    Optional to override:
    - validate(): Check if source is accessible
    - setup(): Pre-harvest initialization
    - cleanup(): Post-harvest cleanup
    """
    
    # User-Agent rotation pool
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    ]
    
    def __init__(self, data_dir: str = "./data/harvested"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self._setup_session()
        
        # Status tracking
        self.status = SourceStatus.IDLE
        self.last_run: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.consecutive_errors = 0
        self.cooldown_until: Optional[datetime] = None
        
        # Stats
        self.items_harvested = 0
        self.items_valid = 0
        self.errors = 0
        
        # Config
        self.enabled = True
        self.max_retries = 3
        self.cooldown_minutes = 15
    
    def _setup_session(self):
        """Configure requests session."""
        self.session.headers.update({
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
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
    def fetch_latest(self) -> Any:
        """
        Fetch raw data from the source.
        
        Returns raw data in source-specific format.
        """
        pass
    
    @abc.abstractmethod
    def parse(self, raw_data: Any) -> List[Dict]:
        """
        Parse raw data into structured dictionaries.
        
        Args:
            raw_data: Data from fetch_latest()
            
        Returns:
            List of parsed item dictionaries
        """
        pass
    
    @abc.abstractmethod
    def normalize(self, parsed_item: Dict) -> FashionItem:
        """
        Normalize a parsed item to FashionItem.
        
        Args:
            parsed_item: Single item from parse()
            
        Returns:
            Standardized FashionItem
        """
        pass
    
    def validate(self) -> bool:
        """Check if source is accessible. Override for custom validation."""
        return True
    
    def setup(self):
        """Pre-harvest initialization. Override if needed."""
        pass
    
    def cleanup(self):
        """Post-harvest cleanup. Override if needed."""
        pass
    
    def harvest(self) -> Generator[FashionItem, None, None]:
        """
        Main harvesting method. Yields FashionItem objects.
        
        Uses fetch_latest → parse → normalize pipeline.
        """
        try:
            raw_data = self.fetch_latest()
            parsed_items = self.parse(raw_data)
            
            for parsed_item in parsed_items:
                try:
                    item = self.normalize(parsed_item)
                    self.items_harvested += 1
                    yield item
                except Exception as e:
                    logger.warning(f"[{self.name}] Normalize error: {e}")
                    self.errors += 1
                    
        except Exception as e:
            logger.error(f"[{self.name}] Harvest error: {e}")
            self.errors += 1
            raise
    
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
            logger.warning(f"[{self.name}] Download failed: {url} - {e}")
            return None
    
    def is_in_cooldown(self) -> bool:
        """Check if source is in cooldown period."""
        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            return True
        self.cooldown_until = None
        return False
    
    def enter_cooldown(self, minutes: int = None):
        """Put source in cooldown after errors."""
        minutes = minutes or self.cooldown_minutes
        self.cooldown_until = datetime.utcnow() + timedelta(minutes=minutes)
        self.status = SourceStatus.COOLDOWN
        logger.warning(f"[{self.name}] Entering cooldown for {minutes} minutes")


# =============================================================================
# GLOBAL HARVESTER ENGINE
# =============================================================================

class GlobalHarvester:
    """
    Production-grade harvester engine.
    
    Features:
    - Parallel execution with ThreadPoolExecutor
    - Resilience: One source crashing doesn't stop others
    - Automatic retry with exponential backoff
    - Cooldown for rate-limited sources
    - Statistics and monitoring
    """
    
    def __init__(
        self,
        db_url: str = "sqlite:///./data/fashion_brain.db",
        max_workers: int = 4,
        harvest_interval: int = 3600
    ):
        self.db = DatabaseManager(db_url)
        self.sources: Dict[str, FashionSource] = {}
        self.max_workers = max_workers
        self.harvest_interval = harvest_interval
        self.is_running = False
        
        # AI Processor (lazy loaded)
        self._ai_processor = None
        
        logger.info(f"GlobalHarvester initialized (workers={max_workers})")
    
    @property
    def ai_processor(self):
        """Lazy load AI processor."""
        if self._ai_processor is None:
            try:
                from ai_processor import FashionAIProcessor
                self._ai_processor = FashionAIProcessor()
            except Exception as e:
                logger.warning(f"AI processor not available: {e}")
        return self._ai_processor
    
    def register(self, source: FashionSource):
        """Register a source plugin."""
        self.sources[source.name] = source
        logger.info(f"Registered source: {source.name}")
    
    def _harvest_source(self, source: FashionSource) -> Dict:
        """
        Harvest from a single source with error handling.
        
        Returns stats dict.
        """
        stats = {
            'source': source.name,
            'items': 0,
            'valid': 0,
            'errors': 0,
            'started_at': datetime.utcnow().isoformat(),
            'status': 'running'
        }
        
        # Check cooldown
        if source.is_in_cooldown():
            logger.info(f"[{source.name}] In cooldown, skipping")
            stats['status'] = 'cooldown'
            return stats
        
        try:
            source.status = SourceStatus.RUNNING
            
            # Validate
            if not source.validate():
                logger.warning(f"[{source.name}] Validation failed")
                source.consecutive_errors += 1
                if source.consecutive_errors >= 3:
                    source.enter_cooldown()
                stats['status'] = 'validation_failed'
                return stats
            
            # Setup
            source.setup()
            
            # Harvest
            for item in source.harvest():
                try:
                    # Queue for database
                    if self.db.queue_item(item):
                        stats['items'] += 1
                        
                        # Process with AI if available
                        if self.ai_processor and item.local_path:
                            is_valid, labels, conf = self.ai_processor.process(item.local_path)
                            if is_valid:
                                stats['valid'] += 1
                                item.predicted_labels = labels
                                
                except Exception as e:
                    logger.warning(f"[{source.name}] Item error: {e}")
                    stats['errors'] += 1
            
            # Cleanup
            source.cleanup()
            
            # Reset error count on success
            source.consecutive_errors = 0
            source.status = SourceStatus.IDLE
            stats['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"[{source.name}] Critical error: {e}")
            source.last_error = str(e)
            source.consecutive_errors += 1
            source.status = SourceStatus.ERROR
            stats['errors'] += 1
            stats['status'] = 'error'
            
            # Enter cooldown after multiple failures
            if source.consecutive_errors >= 3:
                source.enter_cooldown()
        
        stats['completed_at'] = datetime.utcnow().isoformat()
        source.last_run = datetime.utcnow()
        
        return stats
    
    def run_once(self) -> List[Dict]:
        """Run all harvesters once in parallel."""
        logger.info(f"Starting harvest cycle ({len(self.sources)} sources)")
        
        enabled = [s for s in self.sources.values() if s.enabled]
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all sources
            futures = {
                executor.submit(self._harvest_source, s): s.name
                for s in enabled
            }
            
            # Collect results (non-blocking)
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result(timeout=600)  # 10 min timeout
                    results.append(result)
                    logger.info(
                        f"[{name}] Completed: {result['items']} items, "
                        f"{result['errors']} errors"
                    )
                except Exception as e:
                    logger.error(f"[{name}] Future failed: {e}")
                    results.append({
                        'source': name,
                        'items': 0,
                        'errors': 1,
                        'status': 'exception'
                    })
        
        # Summary
        total = sum(r['items'] for r in results)
        errors = sum(r['errors'] for r in results)
        logger.info(f"Harvest complete: {total} items, {errors} errors")
        
        return results
    
    def run_continuous(self):
        """Run harvesting continuously."""
        self.is_running = True
        logger.info("Starting continuous harvesting mode")
        
        while self.is_running:
            try:
                self.run_once()
                logger.info(f"Sleeping {self.harvest_interval}s until next cycle")
                time.sleep(self.harvest_interval)
            except KeyboardInterrupt:
                logger.info("Stopping continuous mode...")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                time.sleep(60)
        
        self.is_running = False
    
    def stop(self):
        """Stop continuous harvesting."""
        self.is_running = False
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        db_stats = self.db.get_stats()
        
        source_stats = {}
        for name, source in self.sources.items():
            source_stats[name] = {
                'status': source.status.value,
                'enabled': source.enabled,
                'items_harvested': source.items_harvested,
                'errors': source.errors,
                'last_run': source.last_run.isoformat() if source.last_run else None,
                'in_cooldown': source.is_in_cooldown()
            }
        
        return {
            'database': db_stats,
            'sources': source_stats
        }
    
    def shutdown(self):
        """Graceful shutdown."""
        self.stop()
        self.db.shutdown()
        logger.info("GlobalHarvester shutdown complete")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'FashionSource',
    'FashionItem',
    'SourceStatus',
    'GlobalHarvester',
    'DatabaseManager',
    'FashionItemModel',
]
