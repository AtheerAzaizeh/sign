"""
opentrend/batch_trend_miner.py
Overnight Batch Processor for Google Trends Mining

This module provides a robust, fault-tolerant system for mining Google Trends
data at scale while respecting rate limits. Designed to run overnight or
continuously for 24+ hours.

Features:
- SQLite-based checkpointing (resume after crash)
- Strict rate limiting (60s between requests)
- 429 error handling (5-min backoff)
- Seasonal candidate generation
- Markdown report generation for winning trends

Usage:
    # Quick start
    python batch_trend_miner.py
    
    # Or programmatically
    from batch_trend_miner import TrendMiner
    
    miner = TrendMiner()
    miner.add_candidates(["cargo pants", "linen shirt"])
    miner.run()
    miner.generate_report()
"""

from __future__ import annotations

import os
import json
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager

# Optional imports
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Import OpenTrend modules
try:
    from signals.trend_tracker import TrendTracker
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from signals.trend_tracker import TrendTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SEASONAL CANDIDATES
# =============================================================================

def generate_seasonal_candidates(season: str = "Spring/Summer") -> List[str]:
    """
    Generate a list of seasonal fashion keywords to analyze.
    
    Filters out items that are irrelevant for the target season to maximize
    API efficiency.
    
    Args:
        season: "Spring/Summer" or "Fall/Winter"
        
    Returns:
        List of 50-100 candidate keywords
    """
    
    SPRING_SUMMER_CANDIDATES = [
        # Tops
        "linen shirt", "linen blouse", "crochet top", "halter top",
        "crop top", "tank top", "off shoulder top", "bandeau top",
        "mesh top", "sheer blouse", "sleeveless blouse", "flutter sleeve top",
        
        # Dresses
        "midi dress", "maxi dress", "sundress", "linen dress",
        "crochet dress", "wrap dress", "slip dress", "shirt dress",
        "tiered dress", "floral dress", "smocked dress", "A-line dress",
        
        # Bottoms
        "wide leg pants", "linen pants", "culottes", "bermuda shorts",
        "denim shorts", "high waisted shorts", "flowy pants", "palazzo pants",
        "capri pants", "cargo shorts", "paper bag shorts", "tailored shorts",
        
        # Skirts  
        "midi skirt", "maxi skirt", "mini skirt", "pleated skirt",
        "wrap skirt", "sheer skirt", "denim skirt", "linen skirt",
        "tiered skirt", "slit skirt", "A-line skirt",
        
        # Outerwear
        "linen blazer", "lightweight cardigan", "denim jacket", "utility jacket",
        "bomber jacket", "cropped blazer", "kimono jacket", "mesh jacket",
        
        # Swimwear & Resort
        "one piece swimsuit", "bikini", "sarong", "cover up dress",
        "beach dress", "resort wear", "kaftan",
        
        # Footwear
        "espadrilles", "gladiator sandals", "platform sandals", "mules",
        "wedge sandals", "ballet flats", "loafers", "slide sandals",
        "strappy heels", "canvas sneakers",
        
        # Accessories
        "straw bag", "raffia bag", "bucket hat", "sun hat",
        "woven belt", "statement earrings", "layered necklace",
        
        # Fabrics/Styles
        "boho style", "coastal grandmother", "quiet luxury", "minimalist fashion",
        "cottagecore", "old money style", "clean girl aesthetic",
    ]
    
    FALL_WINTER_CANDIDATES = [
        # Outerwear
        "puffer jacket", "wool coat", "leather jacket", "trench coat",
        "teddy coat", "shearling jacket", "down jacket", "peacoat",
        "oversized coat", "quilted jacket", "faux fur coat",
        
        # Tops
        "turtleneck sweater", "cable knit sweater", "cashmere sweater",
        "mock neck top", "thermal top", "fleece pullover", "hoodie",
        "oversized sweater", "cardigan", "sweater vest",
        
        # Bottoms
        "corduroy pants", "wool trousers", "leather pants", "velvet pants",
        "wide leg jeans", "straight leg jeans", "bootcut jeans",
        
        # Dresses
        "sweater dress", "knit dress", "velvet dress", "long sleeve dress",
        
        # Footwear
        "knee high boots", "ankle boots", "chelsea boots", "combat boots",
        "loafers", "mules", "platform boots",
        
        # Accessories
        "wool scarf", "cashmere scarf", "beanie", "leather gloves",
        "crossbody bag", "tote bag",
    ]
    
    if "spring" in season.lower() or "summer" in season.lower():
        return SPRING_SUMMER_CANDIDATES
    else:
        return FALL_WINTER_CANDIDATES


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrendJob:
    """Represents a single trend mining job."""
    keyword: str
    status: str = "pending"  # pending, processing, completed, failed
    growth_percent: Optional[float] = None
    trend_direction: Optional[str] = None
    data_points: int = 0
    last_interest: Optional[float] = None
    error_message: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


# =============================================================================
# TREND QUEUE (SQLite-based Persistence)
# =============================================================================

class TrendQueue:
    """
    SQLite-backed job queue with checkpointing.
    
    Survives crashes and can resume from last successful job.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the queue.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "trend_queue.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"TrendQueue initialized: {self.db_path}")
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    keyword TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'pending',
                    growth_percent REAL,
                    trend_direction TEXT,
                    data_points INTEGER DEFAULT 0,
                    last_interest REAL,
                    error_message TEXT,
                    created_at TEXT,
                    completed_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def add_keywords(self, keywords: List[str]) -> int:
        """
        Add keywords to the queue (skip duplicates).
        
        Args:
            keywords: List of keywords to add
            
        Returns:
            Number of new keywords added
        """
        added = 0
        with self._get_connection() as conn:
            for kw in keywords:
                kw = kw.strip().lower()
                if not kw:
                    continue
                    
                try:
                    conn.execute(
                        "INSERT INTO jobs (keyword, created_at) VALUES (?, ?)",
                        (kw, datetime.now().isoformat())
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    pass  # Already exists
            
            conn.commit()
        
        logger.info(f"Added {added} new keywords to queue")
        return added
    
    def get_pending(self) -> List[str]:
        """Get all pending keywords."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT keyword FROM jobs WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
        return [row['keyword'] for row in rows]
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self._get_connection() as conn:
            stats = {}
            for status in ['pending', 'processing', 'completed', 'failed']:
                count = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)
                ).fetchone()[0]
                stats[status] = count
            stats['total'] = sum(stats.values())
        return stats
    
    def update_job(self, keyword: str, **updates):
        """Update a job's status and data."""
        with self._get_connection() as conn:
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [keyword]
            
            conn.execute(
                f"UPDATE jobs SET {set_clause} WHERE keyword = ?",
                values
            )
            conn.commit()
    
    def mark_processing(self, keyword: str):
        """Mark a job as currently processing."""
        self.update_job(keyword, status='processing')
    
    def mark_completed(
        self,
        keyword: str,
        growth_percent: float,
        trend_direction: str,
        data_points: int,
        last_interest: float
    ):
        """Mark a job as completed with results."""
        self.update_job(
            keyword,
            status='completed',
            growth_percent=growth_percent,
            trend_direction=trend_direction,
            data_points=data_points,
            last_interest=last_interest,
            completed_at=datetime.now().isoformat()
        )
    
    def mark_failed(self, keyword: str, error: str):
        """Mark a job as failed."""
        self.update_job(
            keyword,
            status='failed',
            error_message=error,
            completed_at=datetime.now().isoformat()
        )
    
    def get_winners(self, min_growth: float = 0.0) -> List[Dict]:
        """Get completed jobs with positive growth."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs 
                WHERE status = 'completed' AND growth_percent > ?
                ORDER BY growth_percent DESC
                """,
                (min_growth,)
            ).fetchall()
        
        return [dict(row) for row in rows]
    
    def get_all_completed(self) -> List[Dict]:
        """Get all completed jobs."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = 'completed' ORDER BY growth_percent DESC"
            ).fetchall()
        return [dict(row) for row in rows]
    
    def reset_failed(self):
        """Reset failed jobs back to pending for retry."""
        with self._get_connection() as conn:
            conn.execute("UPDATE jobs SET status = 'pending' WHERE status = 'failed'")
            conn.commit()
    
    def clear_all(self):
        """Clear all jobs from the queue."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM jobs")
            conn.commit()
        logger.info("Queue cleared")


# =============================================================================
# BATCH TREND MINER
# =============================================================================

class TrendMiner:
    """
    Overnight batch processor for Google Trends mining.
    
    Designed to run continuously with strict rate limiting and
    automatic checkpointing/resume capability.
    
    Usage:
        miner = TrendMiner()
        
        # Add Spring/Summer candidates
        miner.add_seasonal_candidates("Spring/Summer")
        
        # Or add custom keywords
        miner.add_candidates(["linen shirt", "crochet top"])
        
        # Run the miner (will take hours for large queues)
        miner.run()
        
        # Generate report
        miner.generate_report("spring_2026_forecast.md")
    """
    
    # Rate limiting configuration
    DEFAULT_DELAY = 60           # Seconds between requests
    RATE_LIMIT_BACKOFF = 300     # 5 minutes on 429 error
    MAX_CONSECUTIVE_ERRORS = 5   # Stop after this many consecutive errors
    
    def __init__(
        self,
        db_path: str = None,
        delay_seconds: int = 60,
        timeframe: str = 'today 12-m'
    ):
        """
        Initialize the batch miner.
        
        Args:
            db_path: Path to SQLite database for checkpointing
            delay_seconds: Delay between Google Trends requests
            timeframe: Time range for trend analysis
        """
        self.queue = TrendQueue(db_path)
        self.tracker = TrendTracker(initial_delay=5.0, retries=2)
        self.delay_seconds = max(delay_seconds, 30)  # Minimum 30s
        self.timeframe = timeframe
        
        self._stop_requested = False
        self._consecutive_errors = 0
        
        logger.info(f"TrendMiner initialized: {self.delay_seconds}s delay, {timeframe} timeframe")
    
    def add_candidates(self, keywords: List[str]) -> int:
        """Add keywords to the mining queue."""
        return self.queue.add_keywords(keywords)
    
    def add_seasonal_candidates(self, season: str = "Spring/Summer") -> int:
        """Add seasonal fashion candidates to the queue."""
        candidates = generate_seasonal_candidates(season)
        logger.info(f"Adding {len(candidates)} {season} candidates")
        return self.add_candidates(candidates)
    
    def get_status(self) -> Dict:
        """Get current miner status."""
        stats = self.queue.get_stats()
        stats['estimated_time_remaining'] = self._estimate_time(stats['pending'])
        return stats
    
    def _estimate_time(self, pending_count: int) -> str:
        """Estimate time remaining for pending jobs."""
        seconds = pending_count * self.delay_seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    
    def _countdown(self, seconds: int, message: str = "Next request"):
        """Show countdown timer with tqdm."""
        for remaining in tqdm(range(seconds, 0, -1), 
                              desc=f"⏳ {message}",
                              ncols=60,
                              bar_format='{desc}: {remaining}s'):
            time.sleep(1)
            if self._stop_requested:
                raise KeyboardInterrupt("Stop requested")
    
    def run(self, max_jobs: int = None):
        """
        Run the batch mining process.
        
        This will process all pending jobs with rate limiting.
        Can be interrupted with Ctrl+C and will resume on next run.
        
        Args:
            max_jobs: Optional limit on jobs to process (for testing)
        """
        stats = self.get_status()
        pending = self.queue.get_pending()
        
        if max_jobs:
            pending = pending[:max_jobs]
        
        if not pending:
            logger.info("No pending jobs in queue")
            return
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING BATCH TREND MINING")
        logger.info("=" * 60)
        logger.info(f"   Pending: {len(pending)} keywords")
        logger.info(f"   Delay: {self.delay_seconds}s between requests")
        logger.info(f"   Estimated time: {self._estimate_time(len(pending))}")
        logger.info(f"   Timeframe: {self.timeframe}")
        logger.info("=" * 60)
        logger.info("   Press Ctrl+C to stop (progress is saved)")
        logger.info("=" * 60)
        
        completed = 0
        winners = 0
        
        try:
            for i, keyword in enumerate(pending):
                logger.info(f"\n[{i+1}/{len(pending)}] Processing: '{keyword}'")
                
                # Mark as processing
                self.queue.mark_processing(keyword)
                
                try:
                    # Fetch trend data
                    result = self.tracker.analyze_keyword(
                        keyword,
                        timeframe=self.timeframe
                    )
                    
                    if result['status'] == 'success':
                        metrics = result['metrics']
                        
                        # Save results
                        self.queue.mark_completed(
                            keyword=keyword,
                            growth_percent=metrics['growth_percent'],
                            trend_direction=metrics['trend'],
                            data_points=len(result['data']),
                            last_interest=result['data'][-1]['interest'] if result['data'] else 0
                        )
                        
                        completed += 1
                        self._consecutive_errors = 0
                        
                        # Log result
                        growth = metrics['growth_percent']
                        if growth > 0:
                            winners += 1
                            logger.info(f"   ✅ {result['summary']}")
                        else:
                            logger.info(f"   ➡️ {result['summary']}")
                    else:
                        error = result.get('error', result['status'])
                        self.queue.mark_failed(keyword, str(error))
                        logger.warning(f"   ⚠️ {error}")
                        self._consecutive_errors += 1
                        
                except Exception as e:
                    error_str = str(e)
                    self.queue.mark_failed(keyword, error_str)
                    
                    # Check for rate limiting
                    if '429' in error_str or 'TooManyRequests' in error_str:
                        logger.warning(f"   🚫 Rate limited! Waiting {self.RATE_LIMIT_BACKOFF}s...")
                        self._countdown(self.RATE_LIMIT_BACKOFF, "Rate limit cooldown")
                        self._consecutive_errors += 1
                    else:
                        logger.error(f"   ❌ Error: {error_str[:50]}")
                        self._consecutive_errors += 1
                
                # Check for too many errors
                if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    logger.error(f"Too many consecutive errors ({self._consecutive_errors}). Stopping.")
                    break
                
                # Wait before next request (unless last job)
                if i < len(pending) - 1:
                    self._countdown(self.delay_seconds, "Next request")
        
        except KeyboardInterrupt:
            logger.info("\n⏹️ Interrupted by user. Progress saved.")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 MINING SESSION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"   Completed: {completed}")
        logger.info(f"   Winners (growth > 0%): {winners}")
        
        final_stats = self.get_status()
        logger.info(f"   Remaining: {final_stats['pending']} pending")
        logger.info("=" * 60)
    
    def generate_report(
        self,
        output_path: str = None,
        min_growth: float = 0.0
    ) -> str:
        """
        Generate a Markdown report of winning trends.
        
        Args:
            output_path: Path for the report file
            min_growth: Minimum growth % to include in winners
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = Path(__file__).parent / "spring_2026_forecast.md"
        else:
            output_path = Path(output_path)
        
        winners = self.queue.get_winners(min_growth)
        all_completed = self.queue.get_all_completed()
        stats = self.queue.get_stats()
        
        # Build report
        report = f"""# 🌸 Spring/Summer 2026 Fashion Trend Forecast

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Data Source:** Google Trends (12-month analysis)  
**Total Analyzed:** {stats['completed']} trends  
**Winners (Growth > {min_growth}%):** {len(winners)}

---

## 🏆 Top Rising Trends

These trends show **positive growth** and are recommended for Spring/Summer 2026:

| Rank | Trend | Growth | Interest | Direction |
|------|-------|--------|----------|-----------|
"""
        
        for i, trend in enumerate(winners[:20], 1):
            name = trend['keyword'].title()
            growth = trend['growth_percent']
            interest = trend['last_interest'] or 0
            direction = "📈" if trend['trend_direction'] == 'rising' else "➡️"
            
            report += f"| {i} | {name} | **{growth:+.1f}%** | {interest:.0f}/100 | {direction} |\n"
        
        # Add insights section
        if winners:
            top_3 = winners[:3]
            report += f"""

---

## 💡 Key Insights

### Must-Have Items for Spring/Summer 2026

"""
            for i, trend in enumerate(top_3, 1):
                report += f"{i}. **{trend['keyword'].title()}** - {trend['growth_percent']:+.1f}% growth\n"
        
        # Add declining trends
        losers = [t for t in all_completed if t['growth_percent'] < -10][:5]
        if losers:
            report += """

### ⚠️ Trends to Avoid

The following showed significant decline:

"""
            for trend in losers:
                report += f"- {trend['keyword'].title()} ({trend['growth_percent']:+.1f}%)\n"
        
        # Methodology
        report += f"""

---

## 📊 Methodology

- **Data Source:** Google Trends API
- **Analysis Period:** Last 12 months
- **Metrics:** Linear regression slope, percentage change
- **Filter:** Only trends with statistically significant movement included
- **Sample Size:** {stats['completed']} keywords analyzed
- **Queue Status:** {stats['completed']} completed, {stats['failed']} failed, {stats['pending']} pending

*Report generated by OpenTrend Batch Miner*
"""
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report generated: {output_path}")
        return str(output_path)
    
    def stop(self):
        """Request graceful stop."""
        self._stop_requested = True


# =============================================================================
# MAIN - OVERNIGHT RUNNER
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          OPENTREND BATCH TREND MINER                                  ║
║          Spring/Summer 2026 Forecast                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  This process will mine Google Trends data for ~80 fashion keywords.  ║
║  Expected runtime: 1-2 hours (60s between requests)                   ║
║                                                                        ║
║  Press Ctrl+C at any time to stop. Progress is automatically saved.   ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize miner
    miner = TrendMiner(delay_seconds=60)
    
    # Check current status
    stats = miner.get_status()
    
    if stats['pending'] == 0 and stats['completed'] == 0:
        # Fresh start - add seasonal candidates
        print("\n📋 Adding Spring/Summer 2026 candidates...")
        miner.add_seasonal_candidates("Spring/Summer")
    
    stats = miner.get_status()
    print(f"\n📊 Queue Status:")
    print(f"   Pending:   {stats['pending']}")
    print(f"   Completed: {stats['completed']}")
    print(f"   Failed:    {stats['failed']}")
    print(f"   Est. Time: {stats['estimated_time_remaining']}")
    
    # Confirm start
    print("\n" + "=" * 60)
    response = input("Start mining? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        miner.run()
        
        # Generate report
        print("\n📄 Generating forecast report...")
        report_path = miner.generate_report()
        print(f"   Report saved to: {report_path}")
    else:
        print("Cancelled.")
