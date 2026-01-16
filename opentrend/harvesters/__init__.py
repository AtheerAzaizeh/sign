"""
opentrend/harvesters/__init__.py
Global Fashion Brain - Harvester Plugin Registry

Production-grade plugin system with:
- SQLAlchemy ORM for database abstraction
- Queue-based writes for scalability
- Resilient GlobalHarvester with error isolation
"""

# Core infrastructure
from .core import (
    FashionSource,
    FashionItem,
    SourceStatus,
    GlobalHarvester,
    DatabaseManager,
    FashionItemModel,
)

# Source plugins
from .street_signal import RedditHarvester, PinterestHarvester
from .runway import VogueRunwayHarvester, TheImpressionHarvester
from .visual_vibe import UnsplashHarvester, PexelsHarvester
from .marketplace import DepopHarvester, GrailedHarvester

__all__ = [
    # Core
    'FashionSource',
    'FashionItem',
    'SourceStatus',
    'GlobalHarvester',
    'DatabaseManager',
    'FashionItemModel',
    
    # Type A: Social Demand
    'RedditHarvester',
    'PinterestHarvester',
    
    # Type B: High Fashion Supply
    'VogueRunwayHarvester',
    'TheImpressionHarvester',
    
    # Type C: Visuals
    'UnsplashHarvester',
    'PexelsHarvester',
    
    # Type D: Market Validation
    'DepopHarvester',
    'GrailedHarvester',
]

# Version
__version__ = "1.0.0"
