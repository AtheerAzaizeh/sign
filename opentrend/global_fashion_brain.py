"""
opentrend/global_fashion_brain.py
Main entry point for the Global Fashion Brain

This is the orchestrator that runs the entire automated
fashion data ingestion and learning system.

Usage:
    # Interactive mode
    python global_fashion_brain.py
    
    # Continuous 24/7 mode
    python global_fashion_brain.py --continuous
    
    # Single harvest cycle
    python global_fashion_brain.py --once
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./data/fashion_brain.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
Path('./data').mkdir(exist_ok=True)


def print_banner():
    """Print startup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗ ██╗      ██████╗ ██████╗  █████╗ ██╗         ███████╗ █████╗ ███████║
║  ██╔════╝ ██║     ██╔═══██╗██╔══██╗██╔══██╗██║         ██╔════╝██╔══██╗██╔════║
║  ██║  ███╗██║     ██║   ██║██████╔╝███████║██║         █████╗  ███████║███████║
║  ██║   ██║██║     ██║   ██║██╔══██╗██╔══██║██║         ██╔══╝  ██╔══██║╚════██║
║  ╚██████╔╝███████╗╚██████╔╝██████╔╝██║  ██║███████╗    ██║     ██║  ██║███████║
║   ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝╚══════║
║                                                                               ║
║                    ██████╗ ██████╗  █████╗ ██╗███╗   ██╗                      ║
║                    ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                      ║
║                    ██████╔╝██████╔╝███████║██║██╔██╗ ██║                      ║
║                    ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║                      ║
║                    ██████╔╝██║  ██║██║  ██║██║██║ ╚████║                      ║
║                    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                      ║
║                                                                               ║
║                      🧠 Automated Fashion Intelligence System                 ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Harvesters: Reddit | Pinterest | Vogue | Unsplash | Pexels | Depop | Grailed  ║
║ AI Stack:   CLIP Auto-Labeler | LSTM | Transformer | Prophet | ResNet50       ║
║ Storage:    SQLite + Local Images                                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


def setup_harvesters():
    """Initialize all harvesters."""
    from harvesters import (
        GlobalHarvester,
        RedditHarvester,
        PinterestHarvester,
        VogueRunwayHarvester,
        TheImpressionHarvester,
        UnsplashHarvester,
        PexelsHarvester,
        DepopHarvester,
        GrailedHarvester
    )
    
    harvester = GlobalHarvester(
        db_path="./data/fashion_brain.db",
        max_workers=4,
        harvest_interval=3600  # 1 hour
    )
    
    # Register all sources
    sources = [
        # Street Signal (Social Media)
        ("Reddit", RedditHarvester(
            client_id=os.environ.get('REDDIT_CLIENT_ID'),
            client_secret=os.environ.get('REDDIT_CLIENT_SECRET')
        )),
        ("Pinterest", PinterestHarvester()),
        
        # Runway (High Fashion)
        ("Vogue Runway", VogueRunwayHarvester(max_designers=10)),
        ("The Impression", TheImpressionHarvester()),
        
        # Visual Vibe (Open Source)
        ("Unsplash", UnsplashHarvester(
            api_key=os.environ.get('UNSPLASH_API_KEY')
        )),
        ("Pexels", PexelsHarvester(
            api_key=os.environ.get('PEXELS_API_KEY')
        )),
        
        # Marketplace (Hype)
        ("Depop", DepopHarvester()),
        ("Grailed", GrailedHarvester()),
    ]
    
    for name, source in sources:
        try:
            harvester.register(source)
            logger.info(f"✓ Registered: {name}")
        except Exception as e:
            logger.warning(f"✗ Failed to register {name}: {e}")
    
    return harvester


def run_once(harvester):
    """Run a single harvest cycle."""
    logger.info("Starting single harvest cycle...")
    results = harvester.run_once()
    
    # Print results
    print("\n" + "="*60)
    print("📊 HARVEST CYCLE RESULTS")
    print("="*60)
    
    total_items = 0
    total_errors = 0
    
    for result in results:
        items = result.get('items', 0)
        errors = result.get('errors', 0)
        total_items += items
        total_errors += errors
        
        status = "✓" if errors == 0 else "⚠"
        print(f"  {status} {result['source']}: {items} items, {errors} errors")
    
    print("-"*60)
    print(f"  TOTAL: {total_items} items, {total_errors} errors")
    print("="*60)
    
    return total_items, total_errors


def run_continuous(harvester, with_labeling: bool = True):
    """Run continuous harvesting with optional auto-labeling."""
    logger.info("Starting continuous mode...")
    
    # Start auto-labeler in background
    trainer = None
    if with_labeling:
        try:
            from auto_trainer import AutoTrainer
            trainer = AutoTrainer(
                db_path="./data/fashion_brain.db",
                retrain_threshold=1000
            )
            trainer.start()
            logger.info("Auto-labeler started in background")
        except Exception as e:
            logger.warning(f"Could not start auto-labeler: {e}")
    
    try:
        harvester.run_continuous()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        if trainer:
            trainer.stop()


def show_status(harvester):
    """Show current system status."""
    harvester.print_status()
    
    # Show auto-trainer stats if available
    try:
        from auto_trainer import AutoTrainer
        trainer = AutoTrainer(db_path="./data/fashion_brain.db")
        stats = trainer.get_stats()
        
        print("\n📈 AUTO-LABELER STATUS:")
        print(f"  Images Labeled: {stats['images_labeled_total']}")
        print(f"  Pending: {stats['images_pending']}")
        print(f"  Retrains: {stats['retrains']}")
        print(f"  Threshold: {stats['retrain_threshold']}")
    except Exception as e:
        logger.debug(f"Could not get trainer stats: {e}")


def interactive_mode(harvester):
    """Interactive command mode."""
    print("\n📋 COMMANDS:")
    print("  harvest  - Run single harvest cycle")
    print("  status   - Show current status")
    print("  start    - Start continuous mode")
    print("  sources  - List registered sources")
    print("  quit     - Exit")
    print()
    
    while True:
        try:
            cmd = input("🧠 > ").strip().lower()
            
            if cmd == 'harvest':
                run_once(harvester)
            elif cmd == 'status':
                show_status(harvester)
            elif cmd == 'start':
                run_continuous(harvester)
            elif cmd == 'sources':
                print("Registered sources:")
                for name in harvester.sources:
                    enabled = "✓" if harvester.sources[name].enabled else "✗"
                    print(f"  {enabled} {name}")
            elif cmd in ('quit', 'exit', 'q'):
                print("Goodbye! 👋")
                break
            else:
                print(f"Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Global Fashion Brain - Automated Fashion Intelligence System'
    )
    parser.add_argument('--once', action='store_true', help='Run single harvest cycle')
    parser.add_argument('--continuous', action='store_true', help='Run 24/7 continuous mode')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--no-label', action='store_true', help='Disable auto-labeling')
    
    args = parser.parse_args()
    
    print_banner()
    
    logger.info("Initializing harvesters...")
    harvester = setup_harvesters()
    
    if args.status:
        show_status(harvester)
    elif args.once:
        run_once(harvester)
    elif args.continuous:
        run_continuous(harvester, with_labeling=not args.no_label)
    else:
        interactive_mode(harvester)


if __name__ == "__main__":
    main()
